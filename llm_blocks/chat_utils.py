import dotenv
from langchain.callbacks import get_openai_callback
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Either create a .env file or set the environment variable."
    )


class GenericChain:
    def __init__(
        self, template: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2
    ):
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=temperature,
        )
        prompt = PromptTemplate.from_template(template)
        self.chain = LLMChain(llm=llm, prompt=prompt)
        self.logs = []

    def __call__(self, *args, **kwargs):
        inputs = {}
        if args:
            inputs = {
                key: value
                for key, value in zip(self.chain.prompt.input_variables, args)
            }
        if kwargs:
            inputs.update(kwargs)

        start_time = time.time()
        with get_openai_callback() as cb:
            response = self.chain(inputs)
        response_time = time.time() - start_time

        # Store logs in a list of dictionaries
        log_entry = {
            "inputs": inputs,
            "callback": cb,
            "response": response,
            "response_time": response_time,
        }
        self.logs.append(log_entry)
        return response
