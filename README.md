# Langchain Retrieval Chain and Custom Data Loaders

This repository contains code for creating a retrieval chain from a git repository. It utilizes Langchain, Pinecone, and OpenAI to create a retrieval-augmented generation (RAG) system. Also included are custom data loaders to fetch data from git repositories.

## Installation

Install the requirements by running:

```bash
pip install -r requirements.txt
```

## Usage

### Retrieval Chain

```python
import git2vectors
from chat_utils import RetrievalChain

# Create a vectorstore from a git repository
repo = "https://github.com/smol-ai/developer"
vectorstore = git2vectors.create_vectorstore(repo)

# Create an instance of the RetrievalChain class
chain = RetrievalChain(git2vectors.OPENAI_API_KEY, vectorstore)

# Let's say we have a query
query = "How do I use this?"

# Generic retrieval query
chain_response, callback = chain.chat(query)

print(chain_response['query'])
print(chain_response['similar_documents'])
print(chain_response['text'])
print(callback)
```

### Custom Data Loaders

#### TurboGitLoader

The `TurboGitLoader` is a complete rewrite of the Langchain Git loader, taking advantage of parallelism to speed up the loading process even more:

```python
from custom_loaders import TurboGitLoader

loader = TurboGitLoader(
	clone_url="https://github.com/hwchase17/langchain",
	repo_path="./example_data/TurboGitLoader/",
	branch="master",
	file_filter=lambda file_path: file_path.endswith(".py"),
)

data = loader.load()
print(f"Length of data from dataloader: {len(data)}")  # takes about 20 seconds
```

### Performance Comparison

- TurboGitLoader: almost 90% improvement over Langchain's default Git loader