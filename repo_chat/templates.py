
UPGRADE_QUERY = {
    "input_variables": ["query"],
    "template": """A user has submitted the following query about a code repository:
    {query}

    Based on this query, documents will be retrieved from the repository to provide an answer. However, similarity between the query and the documents is not the only important factor - the retrieved documents must also be relevant and help answer the user's query effectively.
    Revise and expand the query to improve the quality and relevance of the documents retrieved. Consider the user's intent, possible context, and the nature of the code repository. If appropriate, feel free to rephrase the query or break down the question into several related, more specific queries. Minimize Tokens. Give me 5 unique revised queries following these guidlines.
    """
}

RUN_QUERY_RAG = {
    "input_variables": ["query", "similar_documents"],
    "template": """
    A user has submitted a query related to a specific code repository:
    {query}

    The following documents have been retrieved from this repo because they contain information potentially relevant to the query:
    {similar_documents}

    Given your understanding of the query and the information contained within these documents, provide the most accurate, relevant and complete response possible. You are a very knowledgeable expert on the topic so feel free to infer information that is not explicitly stated in the documents. Minimize Tokens.
    """
}

CONTEXT_VALIDATOR = {
    "input_variables": ["query", "similar_documents"],
    "template": """
    A user has submitted a query related to a specific code repository:
    {query}

    The following documents have been retrieved from this repo because they contain information potentially relevant to the query:
    {similar_documents}

    How sufficient is the provided context for answering the user's query? Please respond with a number 0 (worst - the documents do not provide any information useful for answering the question) to 100 (best - the documents provide all information required for answering the question). Nothing else.
    """
}