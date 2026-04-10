import os
import time
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class GenerateOutput():
    def __init__(self, text: List[str]):
        self.text = text


class OpenAIModel:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int = 2048,
        api_key: str = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        retry: int = 10,
        response_format: dict = None,
    ) -> GenerateOutput:

        for i in range(1, retry + 1):
            try:
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "n": num_return_sequences,
                }

                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)

                texto = [choice.message.content for choice in response.choices]
                return GenerateOutput(text=texto)

            except Exception as e:
                print(f"Error: {e}, retrying in {i}s...")
                time.sleep(i)

        raise RuntimeError(f"Failed after {retry} retries")


if __name__ == "__main__":
    model = OpenAIModel(
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=9999
    )

    output = model.generate(
        prompt="Say this is a test. Output in json format with two fields: field1 and field2. Each field should contain a short sentence.",
        num_return_sequences=2,
        retry=3,
        response_format={"type": "json_object"}
    )

    print(output.text)