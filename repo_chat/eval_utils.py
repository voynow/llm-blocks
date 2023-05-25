import concurrent.futures
from repo_chat import templates
from repo_chat.chat_utils import ChainManager, RetrievalChain

class CriticChain:
    def __init__(self):
        self.chain_manager = ChainManager()
        self.chain = self.get_chain("CRITIC")

    def get_chain(self, template_type):
        """Get chain by passing the respective template type"""
        return self.chain_manager.create_chain(**getattr(templates, template_type))

    def score(self, query, response):
        """Score a given response to a query"""
        inputs = {"query": query, "response": response}
        return self.chain(inputs)


class QueryEvaluator:
    def __init__(self, get_vectorstore, query, runs_per_query):
        self.get_vectorstore = get_vectorstore
        self.query = query
        self.runs_per_query = runs_per_query
        self.chainlogs = []

    def score_response(self, chain, query):
        """Score a response using CriticChain"""
        critic = CriticChain()
        response = chain.chat(query)
        score = critic.score(query, response["text"])
        response["score"] = score["text"]
        return response

    def evaluate(self):
        """Evaluate RetrievalChain using CriticChain to score responses"""
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

    def get_scores(self):
        """Return responses from all evaluators"""
        scores = {}
        for query, response in self.responses.items():
            scores[query] = [resp["score"] for resp in response]
        return scores
