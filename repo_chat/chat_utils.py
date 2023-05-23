
import dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import concurrent.futures
import os
from repo_chat import templates


class RetrievalChain:
    """
    A class for chatting with a large language model using document retrieval from a GitHub repo.
    It leverages the langchain library to facilitate the retrieval and chat operations.
    """

    def __init__(self, vectorstore, model_name="gpt-3.5-turbo"):
        """
        Initialize the class with the given vectorstore, model_name.
        It also loads the OPENAI_API_KEY from the .env file.
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.chatlog = []
        dotenv.load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def create_chain(self, input_variables, template, temperature=0.1):
        """ Create an LLMChain given model, input_variables, template and temperature.
        """
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name, 
            temperature=temperature
        )
        return LLMChain(llm=llm, prompt=prompt)

    def get_chain(self, template_type):
        """ Get chain by passing the respective template type.
        """
        return self.create_chain(**getattr(templates, template_type))

    def get_chain_inputs(self, query, k=5):
        """
        Retrieve similar documents from vectorstore based on the given query.
        Prepare the chain inputs using these documents.
        """
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        chain_inputs = {
            "query" : query,
            "similar_documents": [doc.page_content for doc, _ in docs]
        }
        return chain_inputs, docs

    def append_to_chatlog(self, chain_inputs, docs, sufficient_context):
        """ Append chain inputs, docs, context to chatlog.
        """
        self.chatlog.append({
            "chain_inputs": chain_inputs,
            "docs": docs,
            "sufficient_context": sufficient_context
        })

    def process_query(self, query, context_validator, run_query, context_threshold=60):
        """
        Process the given query using the context_validator and run_query chains.
        If sufficient context is present, return the result of run_query.
        """
        chain_inputs, docs = self.get_chain_inputs(query)
        val_resp = context_validator(chain_inputs)['text']
        sufficient_context = int(val_resp.lower())
        self.append_to_chatlog(chain_inputs, docs, sufficient_context)
        if sufficient_context > context_threshold:
            return run_query(chain_inputs)
        else:
            return None

    def iterate_through_queries(self, upgrade_query, query, context_validator, run_query, context_threshold):
        """
        For each query retrieved from the upgrade_query response,
        process the query and return the answer if found.
        """
        upgrade_query_resp = upgrade_query(query)['text']
        for q in upgrade_query_resp.split("\n"):
            refined_query = q.lstrip('0123456789. ')
            answer = self.process_query(
                refined_query, context_validator, run_query, context_threshold)
            if answer is not None:
                return answer
        return None
    
    def manage_workflow(self, query, context_validator, run_query):
        """ Manage workflow for upgrading the query and processing it
        """
        context_threshold = 60
        upgrade_query = self.get_chain('UPGRADE_QUERY')

        while True:
            answer = self.iterate_through_queries(
                upgrade_query, query, context_validator, run_query, context_threshold)
            if answer is not None or context_threshold <= 0:
                break
            context_threshold -= 10
        return answer
            
    def chat(self, query):
        """
        Begin chat with the QA chain
        Initialize context_validator and run_query chains.
        """
        context_validator = self.get_chain('CONTEXT_VALIDATOR')
        run_query = self.get_chain('RUN_QUERY_RAG')
        
        with get_openai_callback() as cb:
            answer = self.process_query(query, context_validator, run_query)
            if answer is not None:
                return answer
            else:
                return self.manage_workflow(query, context_validator, run_query)
            
    def get_chatlog(self):
        """ Return chatlog with data pertaining to chat history
        """
        return [{
            "query": log['chain_inputs']['query'],
            "document_similarity_score": [doc[1] for doc in log['docs']],
            "context_score": log['sufficient_context'],
        } for log in self.chatlog]


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

        Score this response on a scale from 0 (completely irrelevant or incorrect) to 100 (perfectly helpful based on the users query). Return the score only.
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

class QueryEvaluator:
    def __init__(self, get_vectorstore, query, runs_per_query):
        self.get_vectorstore = get_vectorstore
        self.query = query
        self.runs_per_query = runs_per_query
        self.chainlogs = []

    def score_response(self, chain, query):
        """Score a response using CriticChain
        """
        critic = CriticChain()
        response = chain.chat(query)
        score = critic.score(query, response['text'])
        response['score'] = score['text']
        return response

    def evaluate(self):
        """ Evaluate RetrievalChain using CriticChain to score responses
        """
        responses = []
        for _ in range(self.runs_per_query):
            vectorstore = self.get_vectorstore()
            chain = RetrievalChain(vectorstore)
            response = self.score_response(chain, self.query)
            responses.append(response)
            self.chainlogs.append(chain.get_chatlog())
        return responses

class MultiQueryEvaluator:
    def __init__(self, get_vectorstore, queries, runs_per_query):
        self.get_vectorstore = get_vectorstore
        self.queries = queries
        self.runs_per_query = runs_per_query

    def run_query(self, query):
        """Run QueryEvaluator for a given query"""
        response = QueryEvaluator(
            self.get_vectorstore, query, self.runs_per_query).evaluate()
        return query, response

    def evaluate(self):
        """Evaluate multiple queries in parallel"""
        responses = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for query, response in executor.map(self.run_query, self.queries):
                responses[query] = response
        self.responses = responses

    def get_scores(self):
        """Return responses from all evaluators"""
        scores = {}
        for query, response in self.responses.items():
            scores[query] = [resp['score'] for resp in response]
        return scores
