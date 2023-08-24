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


class ExecutionLogger:
    def __init__(self):
        self.logs = []

    def log(self, content, response, response_time):
        self.logs.append(
            {
                "inputs": content,
                "response": response,
                "response_time": response_time,
            }
        )


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
        logger: ExecutionLogger,
        message_handler: MessageHandler,
    ):
        self.config = config
        self.logger = logger
        self.message_handler = message_handler

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
        self.logger.log(content, full_response_content, response_time)
        return full_response_content

    def execute(self, content: str) -> Optional[str]:
        self.message_handler.initialize_messages()
        return self.handle_execution(content)

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    def __init__(
        self,
        template: str,
        config: OpenAIConfig,
        logger: ExecutionLogger,
        message_handler: MessageHandler,
    ):
        super().__init__(config=config, logger=logger, message_handler=message_handler)
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


class BlockRegistry:
    def __init__(self):
        self._creators = {}

    def register(self, block_type: str, creator: callable):
        self._creators[block_type] = creator

    def create(self, block_type: str, *args, **kwargs):
        creator = self._creators.get(block_type)
        if creator is None:
            raise ValueError(f"Unknown block type: {block_type}")
        return creator(*args, **kwargs)


def create_generic_block(*args, **kwargs):
    config = OpenAIConfig(*args, **kwargs)
    logger = ExecutionLogger()
    message_handler = MessageHandler(system_message=kwargs.get("system_message"))
    return Block(config=config, logger=logger, message_handler=message_handler)


def create_template_block(template, *args, **kwargs):
    config = OpenAIConfig(*args, **kwargs)
    logger = ExecutionLogger()
    message_handler = MessageHandler(system_message=kwargs.get("system_message"))
    return TemplateBlock(
        template, config=config, logger=logger, message_handler=message_handler
    )


def create_chat_block(*args, **kwargs):
    config = OpenAIConfig(*args, **kwargs)
    logger = ExecutionLogger()
    message_handler = MessageHandler(system_message=kwargs.get("system_message"))
    return ChatBlock(config=config, logger=logger, message_handler=message_handler)



def create_block_registry():
    block_registry = BlockRegistry()
    block_registry.register("generic", create_generic_block)
    block_registry.register("template", create_template_block)
    block_registry.register("chat", create_chat_block)
    return block_registry

