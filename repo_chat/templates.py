
UPGRADE_QUERY = {
    "input_variables": ["query", "repo"],
    "template": """In the context of some github repository ({repo}), a user has submitted the following query:
    {query}

    Based on this query, documents will be retrieved from the repository to provide an answer. However, similarity between the query and the documents is not the only important factor - the retrieved documents must also be relevant and help answer the user's query effectively.
    Revise and expand the query to improve the quality and relevance of the documents retrieved. Consider the user's intent, possible context, and the nature of the code repository. If appropriate, feel free to rephrase the query or break down the question into several related, more specific queries. Minimize Tokens. Give me 5 unique revised queries following these guidlines.
    """
}

RUN_QUERY_RAG = {
    "input_variables": ["query", "similar_documents", "repo"],
    "template": """You are an expert software engineering assistant. In the context of some github repository ({repo}), a user has submitted the following query:
    {query}

    The following documents have been retrieved from this repo because they contain information potentially relevant to the query:
    {similar_documents}

    Given your understanding of the query and the information contained within these documents, provide the most accurate and relevant response possible. You are a very knowledgeable expert on the topic so feel free to infer information that is not explicitly stated in the documents. Be super concise! Your response must be in .md format. Minimize Tokens by using paraphrasing.
    """
}

CONTEXT_VALIDATOR = {
    "input_variables": ["query", "similar_documents", "repo"],
    "template": """You are an expert software engineering assistant. In the context of some github repository ({repo}), a user has submitted the following query:
    {query}

    The following documents have been retrieved from this repo because they contain information potentially relevant to the query:
    {similar_documents}

    How sufficient is the provided context for answering the user's query? Please respond with a number 0 (worst - the documents do not provide any information useful for answering the question) to 100 (best - the documents provide all information required for answering the question). Nothing else.
    """
}

CRITIC = {
    "input_variables": ["query", "response", "repo"],
    "template": """In the context of some github repository ({repo}), a user has submitted the following query:
    {query}

    The following response has been generated:
    {response}

    Does the resonse answer the users query about some specific code repository? Score this response on a scale from 0 (completely irrelevant or incorrect) to 100 (perfectly helpful based on the users query). Return the numeric score only with no words.
    """
}

RAW_CODE = {
    "input_variables": ["query", "repo_data"],
    "template": """Here is my code repository:
    {repo_data}

    Regarding the code above:
    {query}
    """
}