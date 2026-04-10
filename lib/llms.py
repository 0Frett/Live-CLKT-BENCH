import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os
import json


class LanguageModel:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        # Load model (supports LoRA if adapter_config.json is present)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # Try to load base model name from adapter_config.json (if it exists)
        self.base_model_id = None
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                self.base_model_id = config.get("base_model_name_or_path", None)

        print(f"[INFO] Loaded model from: {model_dir}")
        if self.base_model_id:
            print(f"[INFO] Base model: {self.base_model_id}")

    def format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )


    def generate(
        self, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 2
    ) -> list[str]:
        input_text = self.format_prompt(prompt)
        # inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        assert input_ids.shape[0] == 1

        with torch.no_grad():
            if temperature == 0.0:
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        prompt_len = input_ids.shape[-1]
        outputs = [
            self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in output_ids
        ]
        return outputs

