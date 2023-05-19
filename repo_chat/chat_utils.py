
import dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import multiprocessing
import os
from repo_chat import git2vectors
import tiktoken


class RetrievalChain:
    
    def __init__(self, vectorstore, model_name="gpt-3.5-turbo", upgrade_template=None):
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.upgrade_template = upgrade_template

        dotenv.load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")


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

    def upgrade_query(self, query, upgrade_template):
        """Upgrade query to produce better retrieval results"""
        input_variables = ["query"]
        upgrade_query_chain = self.create_chain(input_variables, upgrade_template)
        upgrade_query_chain_inputs = {"query" : query}
        # TODO: add callback
        query = upgrade_query_chain(upgrade_query_chain_inputs)['text']
        return query

    def construct_RAG_chain(self):
        """Construct chain from template and llm"""
        input_variables = ["query", "similar_documents"]
        template = """
        A user has submitted a query related to a specific code repository:
        {query}

        The following documents have been retrieved because they contain information potentially relevant to the query:
        {similar_documents}

        Given your understanding of the query and the information contained within these documents, provide the most accurate, relevant and complete response possible. If the documents don't fully answer the query, you can infer and make educated guesses based on the context provided, just make sure to tell the user.
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
        if self.upgrade_template:
            query = self.upgrade_query(query, self.upgrade_template)
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
        chain_resp['similarity_scores'] = scores

        return chain_resp
    

class CriticChain:

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        dotenv.load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.chain = self.create_chain()

    def create_chain(self):
        """Create chain for scoring responses"""
        input_variables = ["query", "response"]
        template = """
        A user has submitted the following query:
        {query}

        The following response has been generated:
        {response}

        Score this response on a scale from 0 (completely irrelevant or incorrect) to 100 (perfectly answers the query). Return the score only.
        """
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name, temperature=0.1)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def score(self, query, response):
        """Score a given response to a query"""
        inputs = {
            "query": query,
            "response": response
        }
        return self.chain(inputs)



class RetrievalChainEvaluator:
    def __init__(self, query, runs_per_query, templates):
        self.query = query
        self.runs_per_query = runs_per_query
        self.templates = templates
        self.critic = CriticChain()

    def _evaluate_template(self, upgrade_template):
        """Evaluate a single template"""
        vectorstore = git2vectors.get_vectorstore()
        chain = RetrievalChain(vectorstore, upgrade_template=upgrade_template)

        responses = []
        response_scores = []

        for _ in range(self.runs_per_query):
            response = chain.chat(self.query)
            score = self.critic.score(self.query, response['text'])
            responses.append(response)
            response_scores.append(score['text'])

        return {
            'response_scores': response_scores,
            # 'mean_response_score': sum(response_scores) / len(response_scores),
            'responses': responses
        }

    def evaluate(self):
        """Evaluate all templates"""
        with multiprocessing.Pool() as pool:
            results = pool.map(self._evaluate_template, self.templates)
        return results