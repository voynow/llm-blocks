# LLM Blocks :building_construction:

![GitHub stars](https://img.shields.io/github/stars/voynow/llm-blocks?style=social)
![PyPI](https://img.shields.io/pypi/v/llm-blocks)

LLM Blocks is a Python package that provides a simple interface for creating and managing Language Model (LLM) chains. It leverages the power of OpenAI's GPT-3.5-turbo to generate AI completions based on user-defined templates.

## :book: Table of Contents

- [Why Use LLM Blocks](#why-use-llm-blocks)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## :question: Why Use LLM Blocks

LLM Blocks is designed to simplify the process of creating and managing LLM chains. It allows you to define a template and generate AI completions based on that template. This can be particularly useful for tasks such as generating text, answering questions, or creating conversational agents. With LLM Blocks, you can focus on defining your templates and let the package handle the rest.

## :file_folder: Repo Structure

The repository has the following structure:

```
.
├── .gitignore
├── llm_blocks
│   ├── blocks.py
│   └── __init__.py
├── requirements.txt
├── setup.py
├── test.ipynb
└── turbo_docs.toml
```

## :wrench: Installation

You can install LLM Blocks from PyPI:

```bash
pip install llm-blocks
```

## :computer: Usage

Here's a basic example of how to use LLM Blocks:

```python
from llm_blocks import blocks

# Define a template
template = "You're a sophisticated software development AI expert system, capable of assistance with the development of advanced software applications. Your job is to produce comprehensive software architecture designs for MVP software solutions.\\n", "{application_description}"

# Create a block
block = blocks.Block(template=template, stream=True)

# Generate a completion
block(application_description="AI assisted meal planning & grocery list given nutritional goals and dietary restrictions.")
```

In this example, we define a template and use it to create a block. We then generate a completion by calling the block with the `application_description` argument.

## :handshake: Contributing

Contributions are welcome! Please feel free to submit a pull request.

## :email: Contact

If you have any questions or feedback, please reach out to us at voynow99@gmail.com.