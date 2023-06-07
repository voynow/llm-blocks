# LLM Blocks

LLM Blocks is a Python module that helps users interact with Language Learning Model (LLM) chains. It provides a simple and flexible way to create and manage LLM chains, ensuring efficient interactions with models such as OpenAI's GPT-3.5 Turbo.

## Installation
First, make sure to have Python installed. Then, to install the required dependencies for this module, run the following commands:

```
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

## Configuration
To run the LLM Blocks, you'll need an OpenAI API key. Store your API key in a .env file or export it as an environment variable:

For a .env file:
```
OPENAI_API_KEY=your_openai_api_key
```

For environment variables:
```
export OPENAI_API_KEY=your_openai_api_key
```

## Usage
The `llm_blocks` folder contains the main ChatUtils class, which can be utilized to create and manage your LLM chains.

Here's a simple example of using the `GenericChain` class:

```python
from llm_blocks.chat_utils import GenericChain

# Create a chain with a given template
template = "The meaning of {word} is:"
my_chain = GenericChain(template)

# Call the chain with any input you desire
response = my_chain("friendship")
print(response)
```

## Module Structure
- `exclude.toml`: Configuration file to specify files or directories to exclude.
- `requirements.dev.txt`: Development dependencies for this module.
- `requirements.txt`: Main dependencies for this module.
- `llm_blocks`
  - `chat_utils.py`: Python file containing the definition of the `GenericChain` class and utility functions for working with LLM chains.

## Logging Responses
The `GenericChain` class keeps a log of all interactions with the given LLM. The log is stored as a list of dictionaries and can be accessed with `my_chain.logs`.

Example:
```python
for log in my_chain.logs:
    print(f'Inputs: {log["inputs"]}')
    print(f'Callback: {log["callback"]}')
    print(f'Response: {log["response"]}')
    print(f'Response Time: {log["response_time"]}')
    print('---')
```

## Contributing
Feel free to submit pull requests, report bugs, or suggest new features through the GitHub repository. We appreciate your contributions and feedback!

## License
This project is licensed under the MIT License.