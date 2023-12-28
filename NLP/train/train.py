import random
import json
import os

import fire
# import wandb
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification, AutoConfig
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from train.util.dataset import ChatDataset
from train.util.dl import set_random_seed, fix_tokenizer, fix_model
from train.util.io import read_jsonl
from train.util.load import load_saiga

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainerNoBaseSave(Trainer):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        print("Running custom _save_checkpoint")
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


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
    config_file: str,  # trainer_config
    train_file: str,
    val_file: str,
    output_dir: str,
    model_path: str = None,  # Путь до модели и параметров конфигураций.
    adapter_path: str = None,  
    checkpoint: str = False,
    sample_rate: float = 1.0,
    report_to: str = None,
    seed: int = 42,
    use_flash_attention_2: bool = False
):
    set_random_seed(seed)
    # logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    device_map = "auto"
    # ------------------------- DeepSpeed ---------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # Проверяем сколько видеокарт установлено на компьютере. 
    ddp = world_size != 1

    deepspeed_config = config.get("deepspeed")
    trainer_config = config.get("trainer")
    lora_config = config.get("lora")
    callbacks = [SavePeftModelCallback] if lora_config else []

    training_args = TrainingArguments(  # важный параметр.
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        # ddp_find_unused_parameters=False if ddp else None,
        # deepspeed=deepspeed_config,
        **trainer_config
    )

    # training_args = TrainingArguments(
    #     output_dir="./gpt2-sv", #The output directory
    #     overwrite_output_dir=True, #overwrite the content of the output directory
    #     num_train_epochs=3, # number of training epochs
    #     per_device_train_batch_size=4, # batch size for training
    #     per_device_eval_batch_size=4,  # batch size for evaluation
    #     eval_steps = 400, # Number of update steps between two evaluations.
    #     save_steps=800, # after # steps model is saved
    #     warmup_steps=500,# number of warmup steps for learning rate scheduler
    #     )
    
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
    tokenizer.save_pretrained(output_dir)

    # model_type = config.get("model_type", "causal")
    templates_path = config["templates_path"]  # "internal_prompts/saiga_v2.json",
    only_target_loss = config.get("only_target_loss", True)
    mode = config.get("mode", "chat") 
    assert mode == "chat", "Only chat mode is supported in new versions!"
    # assert model_type == "causal", "Only causal models are supported in new versions!"
    max_tokens_count = config["max_tokens_count"]
    #------------------- Загрузка датасета -------------------------
    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    random.shuffle(train_records)
    # print(train_records[0])

    datasets = []
    for records in (train_records, val_records):
        datasets.append(ChatDataset(  # Откуда его взять? Зачем он вообще создается? и каков формат представления данных?
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

    # print("INPUT_IDS")
    # print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
    # print("MASK")
    # print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
    # print("LABELS")
    # print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])

    # model_types = {"causal": AutoModelForCausalLM,}
    
    # ------------------- Загрузка модели -----------------------

    load_in_8bit = bool(config.get("load_in_8bit", False))  # Изменяем параметры модели для дообучения.
    load_in_4bit = bool(config.get("load_in_4bit", False))  # для дообучения модели нужны свои спытания. 
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
        model = fix_model(model, tokenizer, use_resize=False)  # -----
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
        model = fix_model(model, tokenizer, use_resize=False)  # ------
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = fix_model(model, tokenizer)
    # ------------------------------------------------------------- 

    # Default model generation params
    model.config.num_beams = 5
    model.config.max_length = max_tokens_count

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Загружаем модель,
    # изменяем конфиги так, чтобы они соответствовали тем, которые нужны для дообучения модели.
    # передаем конфиги в скрипт для обучения модели. 

    # Обучение новой модели, дообучение существующей.
    # если переданы данные для LoRa, то загружаем ее.
    if adapter_path:
        model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    torch_dtype=torch_dtype
                    )
    else:     
        if lora_config:  # Здесь надо самим загрузить новую модль LoRa. 
            lora_config = LoraConfig(**lora_config)  # Надо понять чем параметры адаптера для обучения отличаются от параметров для предсказаний.
            model = get_peft_model(model, lora_config)   

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset
    # )
    trainer = TrainerNoBaseSave(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        data_collator=data_collator
    )

    # if trainer_config.get("report_to", "wandb") == "wandb":
    #     wandb.init(project="rulm_self_instruct", name=config_file)

    trainer.train(checkpoint)
    print(f'save model to {output_dir}')
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
