import os
import sys
import json
import torch
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import PeftConfig, PeftModel


class ChatModel():
    def __init__(self):
    # , model_name: str = None,
    # use_4bit: bool = True,
    # torch_compile: bool = False,
    # is_lora: bool = False,
    # ):
        # self.model_name = model_name
        # self.use_4bit = use_4bit
        # self.torch_compile = torch_compile
        self.system_prompt = ""
        self.torch_dtype = None
        # self.is_lora = is_lora
        self.use_flash_attention_2 = False
        self.source_max_length: int = 512,
        self.eos_token_id: int = None
        self.model = None,
        self.tokenizer = None
        self.generation_config = None
        #  = self.load_model(model_name)    

    def load(self, # TODO Предполагается, что модель радотает только с одним адаптрером, если таковой имеется.
             model_name_or_path: str = None,
             system_prompt: str = "",
             use_4bit: bool = True,
             is_lora: bool = False, 
             torch_compile: bool = False
             ):
        
        self.system_prompt = system_prompt 
        print(self.system_prompt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False) #, padding_side='left')
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, do_sample=True)
        print(f'use Lora: {is_lora}')
        print(f'use 4 bit: {use_4bit}')
        if not is_lora:
            if use_4bit:
                quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        # bnb_4bit_compute_dtype=self.torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    ),
                self.m4odel = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    # torch_dtype=self.torch_dtype,
                    load_in_4bit=True,
                    device_map="auto",
                    quantization_config=quantization_config,
                    # use_flash_attention_2=self.use_flash_attention_2
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    load_in_8bit=True,
                    device_map="auto"
                )
            # self.model.eval()
        else:   # return  # model, tokenizer, generation_config
            config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)

            if self.torch_dtype is not None:
                self.torch_dtype = getattr(torch, self.torch_dtype)
            else:
                self.torch_dtype = base_model_config.torch_dtype

            # # if device == "cuda":
            # self.model = AutoModelForCausalLM.from_pretrained(
            #         config.base_model_name_or_path,
            #         torch_dtype=self.torch_dtype,
            #         # load_in_4bit=True,
            #         device_map="auto",
            #         # quantization_config=quantization_config,
            #         use_flash_attention_2=self.use_flash_attention_2
            #     )
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=self.torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    load_in_4bit=True,
                    device_map="auto",
                    quantization_config=quantization_config,
                    use_flash_attention_2=self.use_flash_attention_2
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    load_in_8bit=True,
                    device_map="auto",
                    use_flash_attention_2=self.use_flash_attention_2
                )
            self.model = PeftModel.from_pretrained(
                self.model,
                model_name_or_path, # путь к адаптеру в локальной папке.
                torch_dtype=self.torch_dtype
            )
            # elif device == "cpu":
            #     self.model = AutoModelForCausalLM.from_pretrained(
            #         config.base_model_name_or_path,
            #         device_map={"": device},
            #         low_cpu_mem_usage=True
            #     )
            #     self.model = PeftModel.from_pretrained(
            #         self.model,
            #         model_name_or_path,
            #         device_map={"": device}
            #     )

        self.model.eval()
        if torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        # return  # model, tokenizer, generation_config
    
    def dump_memory():
        pass

    # Необходимо сделать промт для модели.
    def generate(self, prompts: List[str]):
        prompts = f"<s>system\n{self.system_prompt}</s>\n" + \
            f"<s>user\n{prompts}</s>\n" + \
            f"<s>bot\n"
        if self.eos_token_id is not None:
            self.generation_config.eos_token_id = self.eos_token_id

        data = self.tokenizer(
            prompts,
            return_tensors="pt",
            # truncation=True,
            # max_length=self.source_max_length,
            # padding=True
        )
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, generation_config=self.generation_config)
        outputs = []
        for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
            sample_output_ids = sample_output_ids[len(sample_input_ids):]
            sample_output = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
            sample_output = sample_output.replace("</s>", "").strip()
            outputs.append(sample_output)
        return outputs[0]
