# Repo-Chat

Repo-Chat is a retrieval-augmented generation system based on Langchain, Pinecone, and OpenAI. It helps developers answer questions about a code repository by using document similarity and context.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

Here's a step-by-step guide on how to use Repo-Chat.

### 1. Create Vectorstore

Once you have your OpenAI API key, you can create a vectorstore for a specific Git repository.

```python
import git2vectors

repo = "https://github.com/smol-ai/developer"
git2vectors.create_vectorstore(repo)
```

### 2. Load Vectorstore

After creating the vectorstore, you can load it as follows:

```python
vectorstore = git2vectors.get_vectorstore()
```

### 3. Use RetrievalChain

Create a `RetrievalChain` with your OpenAI API key, vectorstore and the optional upgrade parameter (set to `True` or `False`).

```python
from chat_utils import RetrievalChain

chain = RetrievalChain(openai_api_key="YOUR_OPENAI_API_KEY", vectorstore=vectorstore, upgrade=True)
```

### 4. Chat with RetrievalChain

Ask a query to the chain for an answer:

```python
query = "How do I use this?"
response = chain.chat(query)

print(response['query'])
print(response['text'])
print(response['similar_documents'])
print(response['scores'])
print(response['callback'])
```

## Important Files

- **chat_utils.py**: Contains the main `RetrievalChain` class and utilities for creating a chat model.
- **custom_loaders.md**: Describes how to load files from a Git repository using custom loaders such as TurboGitLoader.
- **custom_loaders.py**: Implements custom file loaders like `TurboGitLoader`.
- **example.ipynb**: Demonstrates example usage in a Jupyter notebook.
- **git2vectors.py**: Manages the process of creating, loading, and serving vectorstores based on Langchain, Pinecone, and OpenAI.

## Custom File Loaders

The `TurboGitLoader` enables parallel loading of files from a Git repository to speed up the process:

```python
from custom_loaders import TurboGitLoader

loader = TurboGitLoader(
    clone_url="https://github.com/hwchase17/langchain",
    repo_path="./example_data/TurboGitLoader/",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py"),
)

data = loader.load()
```

Performance-wise, the TurboGitLoader offers almost a 90% improvement in loading times compared to Langchain's default Git loader.