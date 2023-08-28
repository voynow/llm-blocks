# LLM Blocks 🤖

![GitHub stars](https://img.shields.io/github/stars/voynow/llm-blocks?style=social)
![PyPI](https://img.shields.io/pypi/v/llm_blocks)

LLM Blocks is a Python library that provides a flexible and easy-to-use interface for interacting with OpenAI's GPT models. It provides a set of classes and methods to handle different types of interactions with the model, such as chat, template, and streamed responses.

## 📚 Table of Contents
- [Why Use LLM Blocks](#why-use-llm-blocks)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)

## 🚀 Why Use LLM Blocks
LLM Blocks simplifies the process of interacting with OpenAI's GPT models. It provides a set of classes and methods that abstract away the complexity of the underlying API calls, allowing you to focus on what matters most - building your application. Whether you're building a chatbot, a code generator, or any other application that leverages AI, LLM Blocks can help you get there faster.

## 📂 Repo Structure
```
llm_blocks
├── blocks.py
├── block_factory.py
├── __init__.py
├── requirements.dev.txt
tests
└── test_blocks.py
```

## 💻 Installation
To install LLM Blocks, you can use pip:
```bash
pip install llm_blocks
```

## 🎯 Usage
Here's a simple example of how to use LLM Blocks:

```python
from llm_blocks import block_factory

# Create a block
block = block_factory.get('block')

# Execute the block with some content
response = block.execute("Hello, world!")
# or execute like a function
response = block("Hello, world!")

# Print the response
print(response)
```

## 🧪 Testing
To run the tests, navigate to the root directory of the project and run:

```bash
python -m unittest discover tests
```

## 🤝 Contributing
Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## 📝 License
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE.md) file for details.

## 📧 Contact
If you have any questions, feel free to reach out to us at [contact@llmblocks.com](mailto:contact@llmblocks.com).
