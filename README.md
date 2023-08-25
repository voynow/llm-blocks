# LLM-Blocks 📦

![GitHub stars](https://img.shields.io/github/stars/voynow/llm-blocks)
![PyPI](https://img.shields.io/pypi/v/llm-blocks)

LLM-Blocks is a Python library that provides a simple interface for creating and managing Language Learning Model (LLM) chains. It is designed to make it easy to create, manage, and execute blocks of code that interact with OpenAI's GPT-3.5-turbo model.

## 📖 Table of Contents
1. [Why Use LLM-Blocks](#why-use-llm-blocks)
2. [Repo Structure](#repo-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## 🎯 Why Use LLM-Blocks
LLM-Blocks simplifies the process of creating and managing LLM chains. It provides a set of classes and methods that abstract the complexities of interacting with OpenAI's GPT-3.5-turbo model. With LLM-Blocks, you can focus on building your application without worrying about the underlying details of the OpenAI API.

## 🏗️ Repo Structure
```
.
├── .gitignore
├── llm_blocks
│   ├── __init__.py
│   ├── blocks.py
│   └── block_factory.py
├── requirements.txt
├── setup.py
└── turbo_docs.toml
```

## 💻 Installation
To install LLM-Blocks, run the following command in your terminal:
```bash
pip install llm-blocks
```

## 🚀 Usage
Here's a simple example of how to use LLM-Blocks:

```python
from llm_blocks import get

# Create a block
block = get("block")

# Execute the block
response = block("Hello, world!")
print(response)
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a pull request.

## 📜 License
LLM-Blocks is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.