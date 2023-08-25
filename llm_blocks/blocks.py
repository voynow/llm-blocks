import os
import re
import time
from dataclasses import dataclass
from typing import (Any, Dict, Generator, List, Optional, Protocol, TypedDict,
                    Union)

import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

@dataclass
class OpenAIConfig:
    model_name: str = "gpt-3.5-turbo-16k"
    temperature: float = 0.1
    stream: bool = False


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


class CompletionStrategy(Protocol):
    def create_completion(self, block: "Block") -> Generator[Dict[str, Any], None, None]:
        ...

class CompletionParser(Protocol):
    def parse(self, message: Dict[str, Any]) -> str:
        ...

class StreamParser(CompletionParser):
    def parse(self, message: Dict[str, Any]) -> str:
        delta = message["choices"][0]["delta"]
        return delta["content"] if "content" in delta else ""

class BatchParser(CompletionParser):
    def parse(self, message: Dict[str, Any]) -> str:
        return message["choices"][0]["message"]["content"]

class StreamCompletion(CompletionStrategy):
    def create_completion(self, block: "Block") -> Generator[Dict[str, Any], None, None]:
        return openai.ChatCompletion.create(
            model=block.config.model_name,
            messages=block.message_handler.messages,
            temperature=block.config.temperature,
            stream=True,
        )

class BatchCompletion(CompletionStrategy):
    def create_completion(self, block: "Block") -> Generator[Dict[str, Any], None, None]:
        yield openai.ChatCompletion.create(
            model=block.config.model_name,
            messages=block.message_handler.messages,
            temperature=block.config.temperature,
            stream=False,
        )

class Block:
    def __init__(
        self,
        config: OpenAIConfig,
        message_handler: MessageHandler,
        completion_strategy: CompletionStrategy,
        completion_parser: CompletionParser,
    ):
        self.config = config
        self.message_handler = message_handler
        self.completion_strategy = completion_strategy
        self.completion_parser = completion_parser
        self.logs: List[Dict[str, Union[str, float]]] = []

    def create_completion(self) -> Generator[Dict[str, Any], None, None]:
        return self.completion_strategy.create_completion(self)

    def handle_execution(self, content: str) -> str:
        self.message_handler.add_message("user", content)
        response_generator = self.create_completion()
        full_response_content = ""

        for message in response_generator:
            parsed_content = self.completion_parser.parse(message)
            full_response_content += parsed_content

        return full_response_content

    def execute(self, content: str) -> Optional[str]:
        self.message_handler.initialize_messages()
        return self.handle_execution(content)

    def log(self, content, response, response_time):
        self.logs.append(
            {
                "inputs": content,
                "response": response,
                "response_time": response_time,
            }
        )

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    def __init__(self, template: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.template = template
        self.input_variables = re.findall(r"\{(\w+)\}", self.template)

    def format_template(self, inputs: Dict[str, Any]) -> str:
        return self.template.format(**inputs)

    def execute(self, *args: Any, **kwargs: Any) -> Optional[str]:
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        content = self.format_template(inputs)
        return super().execute(content)

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return self.execute(*args, **kwargs)


class ChatBlock(Block):
    def execute(self, content: str) -> Optional[str]:
        return self.handle_execution(content)

    def __call__(self, message: str) -> Optional[str]:
        response = self.execute(message)
        self.message_handler.add_message("assistant", response)
        return response
