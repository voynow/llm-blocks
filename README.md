# Repo Chat

Repo Chat is a tool that utilizes retrieval augmented generation (RAG) using Langchain, Pinecone, and OpenAI to provide answers to user queries about a code repository.

## Dependencies

Install the required dependencies by running:

```
pip install GitPython PyGithub langchain pinecone-client pandas matplotlib numpy ipywidgets uuid
```

## Usage

1. First, set up the vector store using the create_vectorstore function with the Git repository URL:

```python
import git2vectors

repo = "https://github.com/smol-ai/developer"
vectorstore = git2vectors.create_vectorstore(repo)
```

2. Run the chat function with your OpenAI API key, the query, and the vector store to get the response:

```python
import chat_utils

Q = "What is this repo? What is smol-ai? Give an example."
chat_utils.chat(git2vectors.OPENAI_API_KEY, Q, vectorstore)
```

## Key Files and Functions

### chat_utils.py

- create_chain: Create a chain from the template and llm.
- construct_upgrade_query_chain: Upgrade a query to produce better information retrieval results.
- get_upgrade_query_inputs: Get chain inputs from a given query.
- construct_RAG_chain: Construct a RAG chain from a template and llm.
- get_RAG_inputs: Get chain inputs from the vector store.
- chat: Apply a query to the chain and fetch its response.
- num_tokens_from_string: Returns the number of tokens in a text string.

### custom_loaders.py

- FastGitLoader: Loads files from a Git repository into a list of documents.
- TurboGitLoader: Loads files from a Git repository into a list of documents in parallel.

### git2vectors.py

- git_load_wrapper: Load the Git repo data using TurboGitLoader.
- get_text_splitter: Get the document splitter function.
- get_openai_embeddings: Get OpenAI embeddings.
- pinecone_init: Initialize Pinecone database.
- upsert_batch: Upsert batch into Pinecone index.
- process_data: Process data into chunks and embed them.
- embed_and_upsert: Embed and upsert data into Pinecone index.
- create_vectorstore: Create and return a vector store.

### example.ipynb

An example notebook using chat_utils and git2vectors to set up the vector store and chat feature.

## Jupyter dependencies

In order to run the example notebook you may need to install the following packages:

```
pip install jupyter ipython
```

Then, you can run Jupyter Notebook with the following command and open the `example.ipynb` file:

```
jupyter notebook
```

## Example Usage

Here is an example query and the response from the Repo Chat tool:

**Query:**

```
What is this repo? What is smol-ai? Give an example.
```

**Response:**

```
Smol-ai is a prototype of a "junior developer" agent that scaffolds an entire codebase out for you once you give it a product spec. Its purpose is to help developers create code scaffolding prompts in a tight loop with the smol dev. The demo example in `prompt.md` shows the potential of AI-enabled, but still firmly human developer-centric workflow. An example of its usage is running `modal run main.py --prompt "a Chrome extension that, when clicked, opens a small window with a page where you can enter a prompt for reading the currently open page and generating some response from openai"`.
```