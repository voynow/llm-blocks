import os
import re
import time

import dotenv
import openai

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
    ):
        self.template = template
        self.input_variables = self._get_input_variables()
        self.message = {"role": role, "content": None}
        self.model_name = model_name
        self.temperature = temperature
        self.logs = []

    def chain(self, inputs: dict):
        self.message['content'] = self.template.format(**inputs)
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[self.message],
            temperature=self.temperature,
        )
        return response.choices[0]['message']['content']

    def _get_input_variables(self):
        return re.findall(r"\{(\w+)\}", self.template)

    def __call__(self, *args, **kwargs):
        inputs = {}
        if args:
            inputs = {key: value for key, value in zip(self.input_variables, args)}
        if kwargs:
            inputs.update(kwargs)

        start_time = time.time()
        response = self.chain(inputs)
        response_time = time.time() - start_time

        # Store logs in a list of dictionaries
        log_entry = {
            "inputs": inputs,
            "response": response,
            "response_time": response_time,
        }
        self.logs.append(log_entry)
        return response
