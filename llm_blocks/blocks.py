from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, TypedDict, Union, Generator

import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


@dataclass
class OpenAIConfig:
    model_name: str = "gpt-3.5-turbo-16k"
    temperature: float = 0.1


class MessageHandler:
    def __init__(self, system_message: Optional[str] = None):
        self.system_message = system_message
        self.initialize_messages()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def initialize_messages(self):
        self.messages = []
        if self.system_message:
            self.add_message("system", self.system_message)


class Block:
    def __init__(
        self,
        config: OpenAIConfig,
        message_handler: MessageHandler,
    ):
        self.config = config
        self.message_handler = message_handler

    def handle_execution(self, content: str) -> openai.ChatCompletion:
        self.message_handler.add_message("user", content)
        return openai.ChatCompletion.create(
            model=self.config.model_name,
            messages=self.message_handler.messages,
            temperature=self.config.temperature,
            stream=True,
        )

    def execute(self, content: str) -> Optional[str]:
        self.message_handler.initialize_messages()
        return self.handle_execution(content)

    def log(self, content, response):
        self.logs.append(
            {
                "inputs": content,
                "response": response,
            }
        )

    def print_stream(self, response_gen: Generator):
        response = ""

        for message in response_gen:
            delta = message["choices"][0]["delta"]
            content_text = delta["content"] if "content" in delta else ""
            print(content_text, end="", flush=True)
            concatenated_response += content_text

        return response

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    def __init__(
        self,
        template: str,
        config: OpenAIConfig,
        message_handler: MessageHandler,
    ):
        super().__init__(config=config, message_handler=message_handler)
        self.template = template
        self.input_variables = re.findall(r"\{(\w+)\}", self.template)

    def format_template(self, inputs: Dict[str, Any]) -> str:
        return self.template.format(**inputs)

    def execute(self, inputs: Dict[str, Any]) -> Optional[str]:
        content = self.format_template(inputs)
        return super().execute(content)

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[str]:
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        return self.execute(inputs)


class ChatBlock(Block):
    def execute(self, content: str) -> Optional[str]:
        return self.handle_execution(content)

    def __call__(self, message: str) -> Optional[str]:
        response = self.execute(message)
        self.message_handler.add_message("assistant", response)
        return response
