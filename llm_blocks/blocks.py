import os
import re
import time
from typing import Any, Dict, Generator, Union

import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Either create a .env file or set the environment variable."
    )
openai.api_key = OPENAI_API_KEY

from abc import ABC
import re
import openai
import time

class Block(ABC):
    def __init__(
        self,
        template: str,
        role: str = "user",
        model_name: str = "gpt-3.5-turbo-16k",
        temperature: float = 0.1,
        stream: bool = False
    ):
        self.template = template
        self.input_variables = self.get_input_variables()
        self.message = {"role": role, "content": None}
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.logs = []

    def get_input_variables(self):
        return re.findall(r"\{(\w+)\}", self.template)

    def create_completion(self, inputs: dict):
        """Create a GPT completion"""
        self.message["content"] = self.template.format(**inputs)
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[self.message],
            temperature=self.temperature,
            stream=True,
        )
        return response

    def execute(self, inputs):
        start_time = time.time()
        response_generator = self.create_completion(inputs)
        full_response_content = ""
        
        for message in response_generator:
            delta = message["choices"][0]["delta"]
            content_text = delta["content"] if "content" in delta else ""
            full_response_content += content_text
            
            if self.stream:
                print(content_text, end="", flush=True)
        
        self.logs.append({
            "inputs": inputs,
            "response": full_response_content,
            "response_time": time.time() - start_time
        })

        if not self.stream:
            return full_response_content

    def __call__(self, *args, **kwargs):
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        return self.execute(inputs)
