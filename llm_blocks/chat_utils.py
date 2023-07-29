import os
import re
import time

import dotenv
import openai
from IPython.display import Markdown, clear_output, display

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Either create a .env file or set the environment variable."
    )


class GenericChain:
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


    def get_input_variables(self) -> list:
        """Get the input variables from the template"""
        return re.findall(r"\{(\w+)\}", self.template)

    def create_completion(self, inputs: dict) -> str:
        """Create a GPT completion"""
        self.message['content'] = self.template.format(**inputs)
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[self.message],
            temperature=self.temperature,
            stream=self.stream,
        )
        return response

    def stream_output(self, inputs) -> None:
        """Get stream GPT completion - fun to watch"""
        response = self.create_completion(inputs)
        collected_content = ""
        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']  
            collected_content += chunk_message['content'] if 'content' in chunk_message else ''
            clear_output(wait=True)
            display(Markdown(collected_content))
        return collected_content

    def batch_output(self, inputs) -> None:
        """Get batch GPT completion - not as fun to watch"""
        start_time = time.time()
        response = self.create_completion(inputs).choices[0]["message"]["content"]
        response_time = time.time() - start_time
        self.logs.append({
            "inputs": inputs,
            "response": response,
            "response_time": response_time,
        })
        return response

    def __call__(self, *args, **kwargs) -> str:
        """Call the model with the given inputs"""
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)

        if self.stream:
            return self.stream_output(inputs) 
        return self.batch_output(inputs)
