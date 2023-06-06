# LLM Repo Assistant

This is a repository for an AI-driven software engineering assistant that leverages the power of large language models (LLMs) in combination with document retrieval from GitHub repositories. The provided code manages chained AI models and provides a system for scoring answers to queries. The code is based on the "langchain" library and has additional utilities to create, manage, and evaluate AI chains in specific contexts.

## Requirements

To install the requirements for this project, please run

```bash
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

## Repository Overview

- `exclude.toml`: Exclude specific files or directories from the repository.
- `requirements.dev.txt`: Development-related dependencies.
- `requirements.txt`: Main dependencies for running the project.
- `llm_blocks/chain_manager.py`: A module to manage creating and storing chains and OpenAI credentials.
- `llm_blocks/chat_utils.py`: Utilities for chatting with LLM and document retrieval from a GitHub repo.
- `llm_blocks/eval_utils.py`: Utilities to evaluate chains and queries and provide user scores.
- `llm_blocks/templates.py`: Template definitions used in the provided chains.

## Workflow

1. Initialize a custom chain:

    ```python
    from llm_blocks.chain_manager import CustomChain
    chain = CustomChain("UPGRADE_QUERY")
    ```

2. Call the chain with input data:

    ```python
    input_data = {
        "query": "How do I install the dependencies?",
        "repo": "user/repo"
    }
    chain_response = chain(input_data)
    ```

3. Chat with an AI-driven software engineering assistant:

    ```python
    from llm_blocks.eval_utils import RetrievalChain
    chat = RetrievalChain(vectorstore, repo)
    answer = chat.chat("How do I install the dependencies?")
    ```

4. Use raw chain for processing raw code input:

    ```python
    from llm_blocks.chat_utils import RawChain
    raw_code = """
    import pandas as pd
    df = pd.read_csv('data.csv')
    """
    chat = RawChain(raw_code)
    answer = chat.chat("What library is being imported?")
    ```

5. Evaluate chains:

    ```python
    from llm_blocks.eval_utils import MultiQueryEvaluator
    evaluator = MultiQueryEvaluator(get_vectorstore, queries, repo, runs_per_query=5)
    evaluator.evaluate(max_workers=4)
    response_df, chainlogs_df = evaluator.flatten_responses()
    ```

## Customization

You can create new AI chains or modify the existing AI chains in the `llm_blocks/templates.py`. Define input variables and templates for the specific chain you want to create or modify.