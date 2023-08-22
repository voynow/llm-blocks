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
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-16k",
        temperature: float = 0.1,
        stream: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.message: Dict[str, Union[str, None]] = {"role": "user", "content": None}
        self.logs: List[Dict[str, Union[Dict[str, Any], str, float]]] = []

    def create_completion(self, content: str) -> openai.ChatCompletion:
        self.message["content"] = content
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=[self.message],
            temperature=self.temperature,
            stream=True,
        )

    def execute(self, inputs: str) -> Optional[str]:
        """
        Executes a GPT completion based on the given inputs and template.

        If the `stream` attribute is True, the content is printed to the console,
        and the method returns None. If `stream` is False, the content is returned
        as a string without printing to the console.

        :param inputs: Input variables to be substituted into the template
        :return: The response content if `stream` is False, otherwise None
        """
        start_time = time.time()
        response_generator = self.create_completion(inputs)
        full_response_content = ""

        for message in response_generator:
            delta = message["choices"][0]["delta"]
            content_text = delta["content"] if "content" in delta else ""
            full_response_content += content_text

            if self.stream:
                print(content_text, end="", flush=True)

        self.logs.append(
            {
                "inputs": inputs,
                "response": full_response_content,
                "response_time": time.time() - start_time,
            }
        )
        if not self.stream:
            return full_response_content

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    def __init__(
        self,
        template: str,
        model_name: str = "gpt-3.5-turbo-16k",
        temperature: float = 0.1,
        stream: bool = False,
    ):
        super().__init__(model_name, temperature, stream)
        self.template = template
        self.input_variables: List[str] = self.get_input_variables()

    def get_input_variables(self) -> List[str]:
        return re.findall(r"\{(\w+)\}", self.template)

    def execute(self, inputs: Dict[str, Any]) -> Optional[str]:
        content = self.template.format(**inputs)
        return super().execute(content)

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[str]:
        """
        Allows the block to be called as a function, passing in the input variables
        as arguments or keyword arguments.
        """
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)
        return self.execute(inputs)


class ChatBlock:
    def __init__(self, template: str, *args, **kwargs):
        self.template_block = TemplateBlock(template, *args, **kwargs)
        self.block = Block(*args, **kwargs)
        self.initial = True
        self.conversation_history = ""

    def start_conversation(self, inputs: Dict[str, Any]) -> Optional[str]:
        response = self.template_block(inputs)
        self.conversation_history += f"(User)\n{inputs}\n(AI)\n{response}"
        self.initial = False
        return response

    def continue_conversation(self, message: str) -> Optional[str]:
        self.conversation_history += f"\n(User)\n{message}\n(AI)\n"
        response = self.block(self.conversation_history)
        self.conversation_history += response
        return response

    def __call__(self, content: str, initial: bool = None) -> Optional[str]:
        initial = initial if initial is not None else self.initial
        if initial:
            return self.start_conversation(content)
        else:
            return self.continue_conversation(content)

    @property
    def logs(self):
        return self.template_block.logs + self.block.logs
