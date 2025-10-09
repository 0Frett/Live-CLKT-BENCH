import os
import threading
import time
from queue import Queue
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class GenerateOutput():
    def __init__(self, text: List[str]):
        self.text = text

class OpenAIWorker(threading.Thread):
    def __init__(self, queue: Queue, model: str, api_key: str, max_tokens: int, rate_limit_per_min: int):
        threading.Thread.__init__(self)
        self.queue = queue
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.rate_limit_per_min = rate_limit_per_min

    def run(self):
        while True:
            prompt, num_return_sequences, temperature, retry, result_queue, response_format = self.queue.get()
            for i in range(1, retry + 1):
                try:
                    if self.rate_limit_per_min is not None:
                        time.sleep(60 / self.rate_limit_per_min)
 
                    messages = [{"role": "user", "content": prompt}]
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": temperature,
                        "n": num_return_sequences
                    }

                    if response_format:
                        kwargs["response_format"] = response_format

                    response = self.client.chat.completions.create(**kwargs)
                    texto = [choice.message.content for choice in response.choices]
                    result_queue.put(GenerateOutput(text=texto))
                    self.queue.task_done()
                    break
 
                except Exception as e:
                    print(f"An Error Occured: {e}, sleeping for {i} seconds")
                    time.sleep(i)
            else:
                result_queue.put(RuntimeError(f"GPTCompletionModel failed to generate output, even after {retry} tries"))
                self.queue.task_done()


class OpenAIModel_parallel():
    def __init__(self, model: str, temperature:float, max_tokens: int = 2048, num_workers: int = 2):
        self.model = model
        self.max_tokens = max_tokens
        self.queue = Queue()
        self.num_workers = num_workers
        self.workers = []
        self.temperature = temperature

        for _ in range(num_workers):
            api_key = os.getenv("OPENAI_API_KEYS", "").split(',')[_%num_workers]
            worker = OpenAIWorker(self.queue, model, api_key, max_tokens, rate_limit_per_min=9999999)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        retry: int = 10,
        response_format: dict = None,
    ) -> GenerateOutput:
        
        result_queue = Queue()
        self.queue.put(
            (
                prompt, 
                num_return_sequences, 
                self.temperature, 
                retry, result_queue, response_format
            )
        )
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result



if __name__ == "__main__":
    model = OpenAIModel_parallel('gpt-4o-mini', temperature=0.8, max_tokens=9999, num_workers=2)
    output = model.generate(
        prompt="Say this is a test. Output in json format with two fields: field1 and field2. Each field should contain a short sentence.",
        num_return_sequences=2,
        retry=3,
        response_format={"type": "json_object"}
    )
    print(output.text)