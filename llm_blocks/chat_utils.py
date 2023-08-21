from abc import ABC, abstractmethod
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


class Block(ABC):
    def __init__(
        self,
        template: str,
        role: str = "user",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        stream: bool = False,
    ):
        """
        Super simple interface for chat-like GPT completions

        template (str): The template to use for the model
        role (str, optional): The role of the user. Defaults to "user".
        model_name (str, optional): The model name to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The temperature to use. Defaults to 0.2.
        stream (bool, optional): Whether to stream the output or not. Defaults to False.
        """
        self.template = template
        self.input_variables = self.get_input_variables()
        self.message = {"role": role, "content": None}
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.logs = []

    def get_input_variables(self):
        return re.findall(r"\{(\w+)\}", self.template)

    def log(self, request, response, response_time=None):
        log_entry = {
            "inputs": request,
            "response": response,
        }
        if response_time:
            log_entry["response_time"] = response_time
        self.logs.append(log_entry)

    def create_completion(
        self, inputs: Dict[str, Any]
    ) -> Union[Dict[str, Any], Generator]:
        """Create a GPT completion"""
        self.message["content"] = self.template.format(**inputs)
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[self.message],
            temperature=self.temperature,
            stream=self.stream,
        )
        return response

    @abstractmethod
    def execute(self, message: str) -> Union[str, Generator]:
        pass

    @abstractmethod
    def display(self, content) -> None:
        pass

    def __call__(self, *args, **kwargs) -> Union[str, Generator]:
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        return self.execute(inputs)


class StreamBlock(Block):
    def execute(self, message):
        inputs = self.get_input_variables()
        response = self.create_completion(
            {key: value for key, value in zip(inputs, message)}
        )
        self.log(message, response)
        return response

    def display(self, content):
        print("", end="", flush=True)

        for message in content:
            delta = message["choices"][0]["delta"]
            content = delta["content"] if "content" in delta else ""
            print(content, end="", flush=True)


class BatchBlock(Block):
    def execute(self, message):
        start_time = time.time()
        inputs = self.get_input_variables()
        response = self.create_completion(
            {key: value for key, value in zip(inputs, message)}
        ).choices[0]["message"]["content"]
        response_time = time.time() - start_time
        self.log(message, response, response_time)
        return response

    def display(self, content):
        print(content)
