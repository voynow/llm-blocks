import json
from langchain.callbacks import get_openai_callback
from llm_blocks import chain_manager
import time


class RetrievalChain:
    """
    A class for chatting with a large language model using document retrieval from a GitHub repo.
    It leverages the langchain library to facilitate the retrieval and chat operations.
    """

    def __init__(self, vectorstore, repo):
        """
        Initialize the class with the given vectorstore
        """
        self.vectorstore = vectorstore
        self.repo = repo
        self.chainlogs = []

    def log_entry(self, chain_name, input_data, output_data, exec_time):
        """Log the given method, input_data, output_data and execution time"""
        self.chainlogs.append({
            "chain_name": chain_name,
            "input_data": json.dumps(input_data),
            "output_data": output_data['text'],
            "execution_time": exec_time
        })

    def get_chain_inputs(self, query, get_docs=True):
        """
        Retrieve similar documents from vectorstore based on the given query.
        Prepare the chain inputs using these documents.
        """
        chain_inputs = {
            "query": query,
            "repo": self.repo,
        }
        if get_docs:
            docs = self.vectorstore.similarity_search_with_score(query, k=5)
            chain_inputs["similar_documents"] = [doc.page_content for doc, _ in docs]
        return chain_inputs
    
    def call_chain(self, chain, input_data):
        """Call the given chain and log the execution time"""
        start_time = time.time()
        output = chain(input_data)
        exec_time = time.time() - start_time
        self.log_entry(chain.name, input_data, output, exec_time)
        return output

    def process_query(self, query, context_validator, run_query, context_threshold=60):
        """
        Process the given query using the context_validator and run_query chains.
        If sufficient context is present, return the result of run_query.
        """
        chain_inputs = self.get_chain_inputs(query)
        val_resp = self.call_chain(context_validator, chain_inputs)
        sufficient_context = int(val_resp["text"].lower())
        if sufficient_context > context_threshold:
            return self.call_chain(run_query, chain_inputs)
        else:
            return None

    def iterate_through_queries(
        self, upgrade_query, query, context_validator, run_query, context_threshold
    ):
        """
        For each query retrieved from the upgrade_query response,
        process the query and return the answer if found.
        """
        chain_inputs = self.get_chain_inputs(query, get_docs=False)
        upgrade_query_resp = self.call_chain(upgrade_query, chain_inputs)
        for q in upgrade_query_resp["text"].split("\n"):
            refined_query = q.lstrip("0123456789. ")
            answer = self.process_query(
                refined_query, context_validator, run_query, context_threshold
            )
            if answer is not None:
                return answer
        return None

    def manage_workflow(self, query, context_validator, run_query):
        """Manage workflow for upgrading the query and processing it"""
        context_threshold = 60
        upgrade_query = chain_manager.get_chain("UPGRADE_QUERY")

        while True:
            answer = self.iterate_through_queries(
                upgrade_query, query, context_validator, run_query, context_threshold
            )
            if answer is not None or context_threshold <= 0:
                break
            context_threshold -= 10
        return answer

    def chat(self, query):
        """
        Begin chat with the QA chain
        Initialize context_validator and run_query chains.
        """
        context_validator = chain_manager.get_chain("CONTEXT_VALIDATOR")
        run_query = chain_manager.get_chain("RUN_QUERY_RAG")

        with get_openai_callback() as cb:
            answer = self.process_query(query, context_validator, run_query)
            if answer is not None:
                return answer
            else:
                return self.manage_workflow(query, context_validator, run_query)


class RawChain:
    """ Chain specific for raw code input """
    def __init__(self, repo_data):
        """ Initialize the basic chain object """
        self.repo_data = repo_data

    def chat(self, query):
        """ Chat with the basic chain object """
        raw_chain = chain_manager.get_chain("RAW_CODE")

        with get_openai_callback() as cb:
            chain_inputs = {
                "query": query,
                "repo_data": self.repo_data,
            }
            output = raw_chain(chain_inputs)
            return output
