from typing import Any
import dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from repo_chat import templates

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class CustomChain:
    """
    A class to manage creating and storing chains/openai credentials.
    """

    def __init__(self, name, temperature=0.1, model_name = "gpt-3.5-turbo"):
        """
        Initialize the class by loading the OPENAI_API_KEY from the .env file.
        """
        self.name = name
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = OPENAI_API_KEY
        self.chain = self.create_chain(**getattr(templates, name))

    def create_chain(self, input_variables, template):
        """Create an LLMChain given model, input_variables, template and temperature."""
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        return LLMChain(llm=llm, prompt=prompt)

    def __call__(self, input_data: Any):
        """Call the chain with the given input_data."""
        return self.chain(input_data)
    

def get_chain(template_type):
    """Get chain by passing the respective template type."""
    return CustomChain(template_type)