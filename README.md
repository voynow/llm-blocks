# LLM-Blocks :building_construction:

![GitHub stars](https://img.shields.io/github/stars/voynow/llm-blocks?style=social) ![PyPI](https://img.shields.io/pypi/v/llm-blocks)

LLM-Blocks is a Python library that provides a simple interface for creating and managing Language Learning Model (LLM) chains. It is designed to make it easy to interact with OpenAI's GPT-3.5-turbo model, allowing you to create completions and generate responses in a chat-like format.

## :book: Table of Contents

- [Why Use LLM-Blocks](#why-use-llm-blocks)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## :question: Why Use LLM-Blocks

LLM-Blocks is designed to simplify the process of creating and managing LLM chains. It provides a high-level interface that abstracts away the complexities of interacting with the OpenAI API, allowing you to focus on creating engaging and interactive chat experiences. Whether you're building a chatbot, a virtual assistant, or any other application that requires conversational AI, LLM-Blocks can help you get there faster.

## :deciduous_tree: Repo Structure

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

## :gear: Installation

LLM-Blocks can be installed via pip:

```bash
pip install llm-blocks
```

## :rocket: Usage

Here's an example of how to use the `StreamBlock` and `BatchBlock` classes in LLM-Blocks:

```python
from llm_blocks import blocks

template = "Hello, {name}!"

# Create a StreamBlock
block = blocks.StreamBlock(template=template)
response_generator = block(name="World")
block.display(response_generator)

# Create a BatchBlock
# Example of how to use the non-defualt model
block = blocks.BatchBlock(template=template, model_name="gpt-4")
response = block(name="World")
block.display(response)
```

## :handshake: Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
