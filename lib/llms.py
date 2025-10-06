import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import os
import json
import torch._dynamo
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
torch._dynamo.config.cache_size_limit = 128
torch.set_float32_matmul_precision('high')

def _looks_like_gemma(model_dir: str, base_model_id: str | None) -> bool:
    s = (base_model_id or "") + " " + (model_dir or "")
    s = s.lower()
    # Match common Gemma-2 ids: "gemma-2-2b-it", "gemma-2-9b-it", etc.
    return ("gemma" in s) or ("google/gemma" in s)

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

        if "phi" in model_dir.lower():
            self.tokenizer.model_max_length = 2048
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            self.tokenizer.padding_side = "right"

        # Try to load base model name from adapter_config.json (if it exists)
        self.base_model_id = None
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
                self.base_model_id = config.get("base_model_name_or_path", None)

        # Detect Gemma-2 and disable Dynamo to avoid recompile storms
        self.is_gemma = _looks_like_gemma(model_dir, self.base_model_id)
        if self.is_gemma:
            # Safest: turn off Dynamo for this process
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            torch._dynamo.config.suppress_errors = True
            # Gemma benefits from cache in KV; ensure it's on
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = True

        print(f"[INFO] Loaded model from: {model_dir}")
        if self.base_model_id:
            print(f"[INFO] Base model: {self.base_model_id}")
        if self.is_gemma:
            print("[INFO] Gemma detected: TorchDynamo disabled for stable inference.")


    def format_prompt(self, prompt: str) -> str:
        if "bloomz" in self.model_dir.lower():
            return prompt  # vanilla models (e.g., Bloomz, GPT2)
        else:
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


class LanguageModelPretrained:
    def __init__(self, model_id: str = "gpt2"):
        """
        Load a pretrained model from Hugging Face.
        If the model is not cached locally, it will be downloaded automatically.
        """
        self.model_id = model_id

        # Load pretrained model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[INFO] Loaded pretrained model: {model_id}")

    def format_prompt(self, prompt: str) -> str:
        # Default: just return the prompt as-is
        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> list[str]:
        input_text = self.format_prompt(prompt)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[-1]
        outputs = [
            self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in output_ids
        ]
        return outputs



class GPT:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY2", ""))

    def generate(
        self, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> list[str]:
        
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "n": num_return_sequences
        }
        response = self.client.chat.completions.create(**kwargs)
        texts = [choice.message.content for choice in response.choices]

        return texts



if __name__ == "__main__":
    # model_path = "models/opinion-set1_Phi-3.5-mini-instruct/checkpoint-epoch-1"
    # model_path = "models/opinion-set1_Qwen2.5-7B-Instruct/checkpoint-epoch-1"
    # model_path = "models/opinion-set1_bloomz-7b1/checkpoint-epoch-1"
    # model_path = "LLM-Culture-Alignment/models/checkpoint-epoch-1"
    # model_path = "models/opinion-set1_Llama-3.1-8B-Instruct/checkpoint-epoch-1"
    # inferencer = LanguageModel(model_path)

    # prompt = "What are the health benefits of green tea?"
    # response = inferencer.generate(prompt,num_return_sequences=2)
    # print("Response:", response)

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY5", ""))

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "I want to know about elon musk ",
            }
        ],
    )

    print(completion.choices[0].message.content)
