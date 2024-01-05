import os
import gc
import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification, AutoConfig
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from util.dataset import ChatDataset
from util.dl import set_random_seed, fix_tokenizer, fix_model
from util.io import read_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_path = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        checkpoint_folder = os.path.join(args.output_dir, checkpoint_path)
        print(f'checkpoint_model_path {checkpoint_folder}')
        kwargs["model"].save_pretrained(checkpoint_folder)
        return control


def custom_prepare_model_for_int8_training(
    model,
    output_embedding_layer_name="lm_head",
    layer_norm_names=["layer_norm"]
):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)
        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    model.gradient_checkpointing_enable()

    return model


def train( 
    config_file: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    model_path: str = None,  # Путь до модели и параметров конфигураций.
    adapter_path: str = None,  
    checkpoint: str = None,
    sample_rate: float = 1.0,
    report_to: str = "none",
    seed: int = 42,
):
    set_random_seed(seed)
    with open(config_file, "r") as r:
        config = json.load(r)

    device_map = "auto"
    # ------------------------- DeepSpeed ---------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    deepspeed_config = config.get("deepspeed")
    print('deepspeed: ', deepspeed_config)
    print(config["model_name"])
    
    trainer_config = config.get("trainer")
    lora_config = config.get("lora")
    callbacks = [SavePeftModelCallback] if lora_config else []

    training_args = TrainingArguments(
        output_dir=output_dir,
        # save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        # ddp_find_unused_parameters=False if ddp else None,
        # deepspeed=deepspeed_config,
        **trainer_config
    )

    model_name = model_path if model_path else config["model_name"]

    if ddp:  # DistributedDataParallel Организует работу с несколькими видеокартами.
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = trainer_config["gradient_accumulation_steps"]
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        trainer_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    # -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer, model_config)

    templates_path = config["templates_path"]
    only_target_loss = config.get("only_target_loss", True)
    mode = config.get("mode", "chat") 
    assert mode == "chat", "Only chat mode is supported in new versions!"
    assert config['model_type'] == "causal", "Only causal models are supported in new versions!"
    max_tokens_count = config["max_tokens_count"]

    # ------------------- Загрузка датасета -------------------------
    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    random.shuffle(train_records)

    datasets = []
    for records in (train_records, val_records):
        datasets.append(ChatDataset(
            records,
            tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=sample_rate,
            templates_path=templates_path,
            only_target_loss=only_target_loss, 
            # add_global_eos=False
        ))
    train_dataset, val_dataset = datasets
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # ------------------- Загрузка модели -----------------------
    print('Load model')
    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))
    print(f'use 8 bit: {load_in_8bit}')
    print(f'use 4 bit: {load_in_4bit}')
    use_bf16 = bool(trainer_config.get("bf16", False))
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if load_in_8bit:
        assert not load_in_4bit
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            # use_flash_attention_2=use_flash_attention_2
        )
        model = fix_model(model, tokenizer, use_resize=False)
        model = custom_prepare_model_for_int8_training(model)
    elif load_in_4bit:
        assert not load_in_8bit
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype
        )
        model = fix_model(model, tokenizer, use_resize=False)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = fix_model(model, tokenizer)

    # Default model generation params
    model.config.num_beams = 5
    model.config.max_length = max_tokens_count

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    if adapter_path:
        print('Load Adapter')
        model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    torch_dtype=torch_dtype,
                    is_trainable=True,
                    # config=lora_config
                    )
    else:     
        if lora_config:
            print('Create Adapter')
            lora_config = LoraConfig(**lora_config)
            model = get_peft_model(model, lora_config)   

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print('training run.')
    trainer.train(checkpoint)
    print(f'save model to {output_dir}')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)

    # ------------------ Освобождаем ресурсы -----------------------
    model = None
    tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
