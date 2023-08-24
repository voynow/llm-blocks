import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict, Union

import dotenv
import openai

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Either create a .env file or set the environment variable."
    )
openai.api_key = OPENAI_API_KEY


class OpenAIConfig(TypedDict):
    model_name: str
    temperature: float
    stream: bool


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
        self.logs = []

    def create_completion(self) -> openai.ChatCompletion:
        return openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=self.message_handler.messages,
            temperature=self.config["temperature"],
            stream=True,
        )

    def handle_execution(self, content: str) -> str:
        start_time = time.time()
        self.message_handler.add_message("user", content)
        response_generator = self.create_completion()
        full_response_content = ""

        for message in response_generator:
            delta = message["choices"][0]["delta"]
            content_text = delta["content"] if "content" in delta else ""
            full_response_content += content_text

            if self.config["stream"]:
                print(content_text, end="", flush=True)

        response_time = time.time() - start_time
        self.log(content, full_response_content, response_time)
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


block_factories = {}


def register_block_type(type_name):
    def decorator(factory):
        block_factories[type_name] = factory
        return factory

    return decorator


@register_block_type("block")
def create_block(*args, **kwargs):
    return Block(
        config=OpenAIConfig(*args, **kwargs),
        message_handler=MessageHandler(system_message=kwargs.get("system_message")),
    )


@register_block_type("template")
def create_template_block(template, *args, **kwargs):
    return TemplateBlock(
        template,
        config=OpenAIConfig(*args, **kwargs),
        message_handler=MessageHandler(system_message=kwargs.get("system_message")),
    )


@register_block_type("chat")
def create_chat_block(*args, **kwargs):
    return ChatBlock(
        config=OpenAIConfig(*args, **kwargs),
        message_handler=MessageHandler(system_message=kwargs.get("system_message")),
    )


def block_factory(type, *args, **kwargs):
    factory = block_factories.get(type)
    if factory is None:
        raise ValueError(f"Unknown block type: {type}")
    return factory(*args, **kwargs)
