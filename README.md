# LLM Blocks

![GitHub stars](https://img.shields.io/github/stars/voynow/llm-blocks?style=social) ![PyPI](https://img.shields.io/pypi/v/llm-blocks)

LLM Blocks is a Python library that provides a simple interface for creating and managing Language Model (LLM) chains. It is designed to make it easy to interact with OpenAI's GPT-3 API.

## 📚 Table of Contents
- [Why Use LLM Blocks](#why-use-llm-blocks)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Why Use LLM Blocks
LLM Blocks makes it easy to create and manage LLM chains. It provides a simple, intuitive interface for interacting with OpenAI's GPT-3 API. With LLM Blocks, you can focus on building your application without worrying about the intricacies of the API.

## 🏗️ Repo Structure
```
llm_blocks
├── .gitignore
├── llm_blocks
│   ├── __init__.py
│   └── blocks.py
├── requirements.txt
├── setup.py
└── turbo_docs.toml
```

## 📦 Installation
To install LLM Blocks, you can use pip:
```bash
pip install llm-blocks
```

## 🚀 Usage
LLM Blocks provides three main classes: `Block`, `TemplateBlock`, and `ChatBlock`. These classes are the recommended interfaces for calling OpenAI APIs.

Here is an example of how to use them:

```python
from llm_blocks import Block, TemplateBlock, ChatBlock

template = "Hello, {name}! How are you?"

# Create a Block
block = Block()
response = block("Hello, world!")

# Create a TemplateBlock
template_block = TemplateBlock(template)
response = template_block("AI Friend")
# template_block(name="AI Friend") also valid

# Create a ChatBlock
chat_block = ChatBlock(template)
response = chat_block("AI Friend")
response = chat_block("What is the meaning of life?")
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a pull request.

## 📧 Contact
If you have any questions or feedback, please feel free to contact the author at voynow99@gmail.com.