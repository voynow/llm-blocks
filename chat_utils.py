from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import tiktoken


class RetrievalChain:
    def __init__(self, openai_api_key, vectorstore, model_name="gpt-3.5-turbo", upgrade=False):
        self.openai_api_key = openai_api_key
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.upgrade = upgrade

    @staticmethod
    def num_tokens_from_string(string, encoding_name):
        """Returns the number of tokens in a text string"""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def create_chain(self, input_variables, template):
        """Create chain from template and llm"""
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name, temperature=0.1)
        return LLMChain(llm=llm, prompt=prompt)

    def construct_upgrade_query_chain(self):
        """Upgrade query to produce better information retrieval results"""
        input_variables = ["query"]
        template = """You are a coding assistant and a user is asking the following question about a code repository:
        {query}

        The contents of this question will be used to retrieve potentially helpful documents used in answering the question. 
        Edit the query to improve the quality of the documents retrieved. If you don't know just return the original query.
        """
        return self.create_chain(input_variables, template)

    def upgrade_query(self, query):
        """Upgrade query to produce better retrieval results"""
        upgrade_query_chain = self.construct_upgrade_query_chain()
        upgrade_query_chain_inputs = {"query" : query}
        # TODO: add callback
        query = upgrade_query_chain(upgrade_query_chain_inputs)['text']
        return query

    def construct_RAG_chain(self):
        """Construct chain from template and llm"""
        input_variables = ["query", "similar_documents"]
        template = """A user is asking a question about a code repository. Here is there query:
        {query}

        Here are some documents containing similar information to the query:
        {similar_documents}

        Respond with "Inadequate context" if the documents are not helpful. Otherwise, return a response to the query.
        """
        return self.create_chain(input_variables, template)

    def get_RAG_inputs(self, query, k=5):
        """Get chain inputs from vectorstore"""
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        chain_inputs = {
            "query" : query,
            "similar_documents": [doc.page_content for doc, _ in docs]
        }
        return chain_inputs, docs

    def get_retrieval_chain(self, query):
        """Get retrieval chain"""
        if self.upgrade:
            query = self.upgrade_query(query)
        rag_chain = self.construct_RAG_chain()
        rag_chain_inputs, docs = self.get_RAG_inputs(query)
        scores = [d[1] for d in docs]
        return rag_chain, rag_chain_inputs, scores

    def chat(self, query):
        """Apply query to chain"""
        chain, inputs, scores = self.get_retrieval_chain(query)
        with get_openai_callback() as cb:
            chain_resp = chain(inputs)

        chain_resp['callback'] = cb
        chain_resp['scores'] = scores

        return chain_resp
