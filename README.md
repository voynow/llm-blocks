# Langchain-RAG

This repository demonstrates how to use Retrieval Augmented Generation (RAG) with Langchain, Pinecone, and OpenAI.

## Requirements

* Python 3.10
* Install the required packages by running `pip install -r requirements.txt`.

## Usage

This code performs the following steps:

* Retrieves data from a git repository.
* Embeds the text content using OpenAI's text embedding model.
* UpSert the information into Pinecone's index.
* Performs similarity search using Pinecone.
* Executes question-answer retrieval chain.

### Quick Start

1. Initialize Pinecone API Key and OpenAI API Key:

```python
from getpass import getpass
PINECONE_API_KEY = getpass("Pinecone API Key: ")
OPENAI_API_KEY = getpass("OpenAI API Key: ")
```

2. Import `rag_utils`:

```python
import rag_utils
```

3. Create a vectorstore using a github repository:

```python
repo = "https://github.com/hwchase17/langchain"
vectorstore = rag_utils.create_vectorstore(repo)
```

4. Get a retrieval chain instance:

```python
qa = rag_utils.get_retrieval_chain(vectorstore)
```

5. Execute a query and get the response:

```python
query = "How can I use this code?"
response = qa.run(query)
print(f"Query:\n{query}\n\nResponse:\n{response}")
```

***Note:*** Make sure to replace the `OPENAI_API_KEY` and `PINECONE_API_KEY` with your own API keys.

## Modules

`rag_utils.py`:

A utility module containing functions to load data from a git repository, create an OpenAI embeddings instance, initialize Pinecone, process the data, and create a vectorstore and retrieval chain.

`example.ipynb`:

An example Jupyter Notebook demonstrating how to use the RAG functionality.