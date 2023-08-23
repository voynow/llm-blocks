import os
import re
import time
from typing import Any, Dict, List, Union, Optional

import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Either create a .env file or set the environment variable."
    )
openai.api_key = OPENAI_API_KEY

class Block:
    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temperature: float = 0.1, stream: bool = False, system_message: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.logs = []
        self.messages = []
        if system_message:
            self.add_message("system", system_message)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def create_completion(self) -> openai.ChatCompletion:
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            stream=True,
        )

    def execute(self, content: str) -> Optional[str]:
        start_time = time.time()
        self.add_message("user", content)
        response_generator = self.create_completion()
        full_response_content = ""

        for message in response_generator:
            delta = message["choices"][0]["delta"]
            content_text = delta["content"] if "content" in delta else ""
            full_response_content += content_text

            if self.stream:
                print(content_text, end="", flush=True)

        self.logs.append(
            {
                "inputs": content,
                "response": full_response_content,
                "response_time": time.time() - start_time,
            }
        )
        return full_response_content

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    def __init__(self, template: str, *args, system_message: Optional[str] = None, **kwargs):
        super().__init__(*args, system_message=system_message, **kwargs)
        self.template = template
        self.input_variables = re.findall(r"\{(\w+)\}", self.template)

    def execute(self, inputs: Dict[str, Any]) -> Optional[str]:
        content = self.template.format(**inputs)
        return super().execute(content)

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[str]:
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        return self.execute(inputs)


class ChatBlock(Block):
    def __call__(self, message: str) -> Optional[str]:
        response = self.execute(message)
        self.add_message("assistant", response)
        return response
