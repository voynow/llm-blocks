import concurrent.futures
import pandas as pd
from repo_chat.chat_utils import  RetrievalChain
from repo_chat import chain_manager

class CriticChain:
    def __init__(self):
        self.chain = chain_manager.get_chain("CRITIC")

    def score(self, query, response):
        """Score a given response to a query"""
        inputs = {"query": query, "response": response}
        return self.chain(inputs)


class QueryEvaluator:
    def __init__(self, get_vectorstore, query, runs_per_query):
        self.get_vectorstore = get_vectorstore
        self.query = query
        self.runs_per_query = runs_per_query

    def score_response(self, chain, query):
        """Score a response using CriticChain"""
        critic = CriticChain()
        response = chain.chat(query)
        score = critic.score(query, response["text"])
        response['score'] = score['text']

        return {
            'chainlogs': chain.chainlogs,
            'response': response
        }

    def evaluate(self):
        """Evaluate RetrievalChain using CriticChain to score responses"""
        responses = []
        for _ in range(self.runs_per_query):
            vectorstore = self.get_vectorstore()
            chain = RetrievalChain(vectorstore)
            response = self.score_response(chain, self.query)
            responses.append(response)

        return responses


class MultiQueryEvaluator:
    def __init__(self, get_vectorstore, queries, runs_per_query):
        self.get_vectorstore = get_vectorstore
        self.queries = queries
        self.runs_per_query = runs_per_query
        print("MultiQueryEvaluator initialized with the following configurations:")
        print(f"Number of queries: {len(queries)}")
        print(f"Runs per query: {runs_per_query}")
        print(f"Total evaluations: {len(queries) * runs_per_query}")
        print("Parallelized by query using concurrent.futures.ProcessPoolExecutor")

    def run_query(self, query):
        """Run QueryEvaluator for a given query"""
        response = QueryEvaluator(
            self.get_vectorstore, query, self.runs_per_query
        ).evaluate()
        return query, response

    def evaluate(self):
        """Evaluate multiple queries in parallel"""
        responses = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for query, response in executor.map(self.run_query, self.queries):
                responses[query] = response
        self.responses = responses


    def flatten_responses(self):
        """
        Convert responses into two dataframes:
        1. chainlogs_df: chainlogs for each query-response pair
        2. response_df: data from each query-response pair
        """
        chainlogs_dfs = []
        response_dfs = []

        for key, data in self.responses.items():
            for obj in data:

                df_chainlogs = pd.json_normalize(obj['chainlogs'])
                df_chainlogs['query'] = key
                chainlogs_dfs.append(df_chainlogs)

                df_response = pd.DataFrame([obj['response']])
                df_response['similar_documents'] = df_response['similar_documents'].apply(lambda x: ','.join(x))
                df_response['query'] = key
                response_dfs.append(df_response)

        chainlogs_df = pd.concat(chainlogs_dfs, ignore_index=True)
        response_df = pd.concat(response_dfs, ignore_index=True)

        return chainlogs_df, response_df


