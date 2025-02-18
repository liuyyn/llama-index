from dotenv import load_dotenv
import os
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding


load_dotenv()
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = os.environ["QDRANT_PORT"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


# step 1: create a QdrantClient instance
# step 2: initialize the qdrant vector store from qdrant client
# step 3: create a storage context with the vector store
# step 4: load index from the storage context


class RAGService:
    instance = None

    def __new__(cls):
        """
        Singleton pattern to ensure only one instance of the RAGService is created. We do not want to reload the index every time we want to use the RAG.
        This method is called before __init__. If the instance is None, a new instance is created. Otherwise, the existing instance is returned.
        """
        if cls.instance is None:
            cls.instance = super(RAGService, cls).__new__(cls)
            cls.instance.__init__()
        return cls.instance

    def __init__(self):
        # load the qdrant client
        client = qdrant_client.QdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
        #  initialize the qdrant vector store from qdrant client
        vector_store = QdrantVectorStore(client=client, collection_name="attractions")
        # create a storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # set the query and embedding models to cohere
        Settings.llm = Cohere(api_key=COHERE_API_KEY, model="command-r-plus")
        Settings.embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name="embed-english-v3.0",
            input_type="search_query",
        )

        # load your index from stored vectors
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    def get_data(self):
        """ """
        pass

    def query(self, query: str):
        """
        Query the index with the given query string.

        Args:
            query (str): The query string.
        Returns:
            str: The response from the query.

        """
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return response
