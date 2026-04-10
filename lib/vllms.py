from vllm import LLM, SamplingParams
from typing import List, Dict, Union, Optional


class VLLMModel:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int = 8000,
        top_p: float = 0.95,
        max_num_seqs: int = 8,
    ):

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.model_name = model

    def _format_prompts(self, inputs: Union[str, List[str]]) -> List[str]:

        if isinstance(inputs, str):
            inputs = [inputs]

        prompts = []
        for inp in inputs:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": inp}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(prompt)
        return prompts


    def generate(
        self,
        inputs: Union[str, List[str]],
        num_return_sequences: int = 1,
    ) -> List[List[str]]:

        prompts = self._format_prompts(inputs)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=num_return_sequences,
        )
        outputs = self.llm.generate(prompts, sampling_params)

        all_results = []
        for request_output in outputs:
            seqs = [seq.text for seq in request_output.outputs]
            all_results.append(seqs)
        return all_results


