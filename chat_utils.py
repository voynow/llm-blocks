from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import tiktoken


def construct_chain(openai_api_key):
    """ Construct chain from template and llm
    """

    input_variables = ["query", "similar_documents"]
    template = """A user is asking a question about a code repository. Here is there query:
    {query}

    Here are some documents containing similar information to the query:
    {similar_documents}

    If you don't know say "idk" else answer the question.
    """

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template,
    )
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-4',
        temperature=0.2
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain


def get_chain_inputs(vectorstore, query, k=5):
    """ Get chain inputs from vectorstore
    """
    docs = vectorstore.similarity_search_with_score(
        query, k=k
    )
    chain_inputs = {
        "query" : query,
        "similar_documents": [doc.page_content for doc, _ in docs]
    }
    return chain_inputs, docs


def chat(chain, inputs):
    """ Apply query to chain
    """
    with get_openai_callback() as cb:
        chain_resp = chain(inputs)

    if "idk" in chain_resp['text']:
        print("RAG FAILED")
    else:
        print(chain_resp['text'])
    print(cb)


def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
