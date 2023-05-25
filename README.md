# Github -> Retrieval Augmented Generation Example

This is an example of using Langchain, Pinecone, and OpenAI to perform Retrieval Augmented Generation (RAG) on a GitHub repository.

## Getting Started

### Prerequisites

Install the required packages specified in `requirements.txt`:

```
PyGithub
langchain
pinecone-client
pandas
matplotlib
numpy
ipywidgets
uuid
python-dotenv
```

### Setup

1. Create a `.env` file in the repository root folder and add your `OPENAI_API_KEY` and `PINECONE_API_KEY`.

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

2. Run the example notebooks `evaluation.ipynb` and `example.ipynb`, which demonstrate how to use the RetrievalChain and QueryEvaluator classes with the provided Python scripts.

## Files

### evaluation.ipynb

This Jupyter Notebook runs multiple queries using the `MultiQueryEvaluator` class in `repo_chat.eval_utils`. It demonstrates how to evaluate various responses to a set of queries using the `CriticChain`.

### example.ipynb

This Jupyter Notebook demonstrates how to use the `RetrievalChain` class in `repo_chat.chat_utils` to chat with a large language model using document retrieval from a GitHub repo.

### requirements.txt

Lists the required Python packages for this project.

### repo_chat/chat_utils.py

Contains the `RetrievalChain`, `ChainManager`, and `CriticChain` classes, which are responsible for managing the chat workflow using Langchain and OpenAI.

### repo_chat/custom_loaders.py

Contains the `TurboGitLoader` class, which is responsible for loading files from a Git repository into a list of documents.

### repo_chat/eval_utils.py

Contains the `QueryEvaluator` and `MultiQueryEvaluator` classes, which are responsible for evaluating query responses using the CriticChain.

### repo_chat/git2vectors.py

This script demonstrates how to use Langchain, Pinecone, and OpenAI to perform Retrieval Augmented Generation (RAG) on a GitHub repository.

## Usage

1. Add your OpenAI API key and Pinecone API key to the .env file.
2. Run the example.ipynb notebook to see a basic usage of the RetrievalChain class.
3. Run the evaluation.ipynb notebook to evaluate multiple queries with the MultiQueryEvaluator class.