# LLM-Blocks :chains:

[![GitHub stars](https://img.shields.io/github/stars/voynow/llm-blocks.svg)](https://github.com/voynow/llm-blocks/stargazers)
[![PyPI version](https://badge.fury.io/py/llm-blocks.svg)](https://pypi.org/project/llm-blocks/)

LLM-Blocks is a Python library that provides a simple interface for creating and managing Language Learning Model (LLM) chains. It leverages the power of OpenAI's GPT-3.5-turbo to generate chat-like completions.

## :book: Table of Contents
- [Why Use LLM-Blocks](#why-use-llm-blocks)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Example Usage](#example-usage)

## :rocket: Why Use LLM-Blocks

LLM-Blocks stands out from the crowd by providing a super simple interface for creating and managing LLM chains. It's perfect for developers who want to leverage the power of GPT-3.5-turbo without getting into the complexities of managing the model. With LLM-Blocks, you can create GPT completions and stream or batch outputs with ease. 

## :file_folder: Repo Structure

```
.
├── .gitignore
├── .env
├── llm_blocks
│   ├── chat_utils.py
│   └── __init__.py
├── requirements.txt
├── setup.py
└── turbo_docs.toml
```

## :wrench: Installation

To install LLM-Blocks, run the following command:

```bash
pip install llm-blocks
```

## :computer: Example Usage

Here's a simple example of how to use the `GenericChain` class in LLM-Blocks:

```python
from llm_blocks.chat_utils import GenericChain

# Initialize the GenericChain class
chain = GenericChain(template="Hello, {name}!")

# Call the model with the given inputs
response = chain(name="John Doe")

# Print the response
print(response)
```

In this example, the `GenericChain` class is initialized with a template. The model is then called with the given inputs, and the response is printed.

## :heart: Support

If you like this project, please give it a :star: on [GitHub](https://github.com/voynow/llm-blocks)!