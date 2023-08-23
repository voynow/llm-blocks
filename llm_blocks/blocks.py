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
    """
    A class representing a reusable block of code for invoking GPT models via the OpenAI API.

    :param model_name: The name of the GPT model to be used
    :param temperature: The temperature for the completion, controlling randomness
    :param stream: Whether to stream the content directly to the console
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-16k",
        temperature: float = 0.1,
        stream: bool = False,
        system_message: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        self.system_message = system_message
        self.logs: List[Dict[str, Union[Dict[str, Any], str, float]]] = []

    def create_completion(self, content: str) -> openai.ChatCompletion:
        """
        Creates a GPT completion based on the given content.

        :param content: The content to be completed
        :return: The response content
        """
        messages = [{"role": "user", "content": content}]
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
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
        return full_response_content

    def __call__(self, content: str) -> Optional[str]:
        return self.execute(content)


class TemplateBlock(Block):
    """
    Extends the Block class to allow for templating of the input content.

    :param template: The template to be completed
    :param model_name: The name of the GPT model to be used
    :param temperature: The temperature for the completion, controlling randomness
    :param stream: Whether to stream the content directly to the console
    """

    def __init__(
        self, template: str, *args, system_message: Optional[str] = None, **kwargs
    ):
        super().__init__(*args, system_message=system_message, **kwargs)
        self.template = template
        self.input_variables: List[str] = self.get_input_variables()

    def get_input_variables(self) -> List[str]:
        return re.findall(r"\{(\w+)\}", self.template)

    def execute(self, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Executes a GPT completion based on the given inputs and template.

        :param inputs: Input variables to be substituted into the template
        :return: The response content if `stream` is False, otherwise None
        """
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
    """
    Implements a chat style conversation using the TemplateBlock and Block classes

    :param template: The template to be completed
    :param model_name: The name of the GPT model to be used
    :param temperature: The temperature for the completion, controlling randomness
    :param stream: Whether to stream the content directly to the console
    """

    def __init__(
        self, template: str, *args, system_message: Optional[str] = None, **kwargs
    ):
        self.template_block = TemplateBlock(
            template, *args, system_message=system_message, **kwargs
        )
        self.block = Block(*args, system_message=system_message, **kwargs)
        self.initial = True
        self.conversation_history = ""

    def start_conversation(self, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Starts a new conversation using the given inputs with the predefined template.

        :param inputs: A dictionary containing key-value pairs for the initial conversation input
        :return: The response content if `stream` is False, otherwise None
        """
        response = self.template_block(inputs)
        self.conversation_history += f"(User)\n{inputs}\n(AI)\n{response}"
        self.initial = False
        return response

    def continue_conversation(self, message: str) -> Optional[str]:
        """
        Continues an existing conversation with the given message.

        :param message: The user's message to continue the conversation
        :return: The response content if `stream` is False, otherwise None
        """
        self.conversation_history += f"\n(User)\n{message}\n(AI)\n"
        response = self.block(self.conversation_history)
        self.conversation_history += response
        return response

    def __call__(self, content: str, initial: bool = None) -> Optional[str]:
        """
        Allows the block to be called as a function, passing in the input variables
        as arguments or keyword arguments.

        :param content: The user's message to continue the conversation
        :param initial: Whether to start a new conversation
        :return: The response content if `stream` is False, otherwise None
        """
        initial = initial if initial is not None else self.initial
        if initial:
            return self.start_conversation(content)
        else:
            return self.continue_conversation(content)

    @property
    def logs(self):
        return self.template_block.logs + self.block.logs
