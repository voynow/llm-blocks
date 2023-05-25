# README

## Overview

This repository demonstrates a Python implementation of Retrieval Augmented Generation (RAG) using Langchain, Pinecone, and OpenAI's Text-Embedding-ADA-002 and gpt-3.5-turbo models. The provided example.ipynb Jupyter notebook can be used to interact with the implementation.

## Requirements

To install the required packages, run the following command:

```sh
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository to your local machine.
2. Open the example.ipynb notebook in Jupyter.
3. Follow the instructions provided in the notebook to interact with the chatbot.

### Create vectorstore

```python
# Create vectorstore, this will take a while
repo = "https://github.com/smol-ai/developer"
git2vectors.create_vectorstore(repo)

# Load vectorstore, this is fast
vectorstore = git2vectors.get_vectorstore()
```
  
### Sample Chatbot Usage

```python
# Create an instance of the class
chain = RetrievalChain(vectorstore)

# Let's say we have a query
query = "Give me a cool use case for this library - create the prompt file to generate this use case."

# Generic retrieval query
response = chain.chat(query)
Markdown(response['text'])
```

### repo_chat Layout

- `chain_manager.py`: A class to manage creating and storing chains/OpenAI credentials.
- `chat_utils.py`: A class for managing interactions with a large language model using document retrieval from a GitHub repo.
- `custom_loaders.py`: Custom loader for fetching files from a Git repository into a list of documents.
- `eval_utils.py`: Classes for evaluating query responses using the `CriticChain`, `QueryEvaluator`, and `MultiQueryEvaluator`.
- `git2vectors.py`: Main script for creating and getting the Pinecone vectorstore from a Git repository.

## Customization & Future Directions

1. Optimizing prompts for high performance in latency and response quality
2. Creating an interface for users to interact with this system outside of jupyter
3. And more!

Please feel free to explore the code, experiment with different settings, and customize the implementation to suit your requirements.