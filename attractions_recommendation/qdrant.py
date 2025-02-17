from dotenv import load_dotenv
import os
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex


load_dotenv()
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = os.environ["QDRANT_PORT"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]


class RAGService:
    def __init__(self):
        self.client = qdrant_client.QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            # set API KEY for Qdrant Cloud
            api_key=QDRANT_API_KEY,
        )
        self.index = None
        self._initialized = False


vector_store = QdrantVectorStore(client=self.client, collection_name="attractions")


storage_context = StorageContext.from_defaults(vector_store=vector_store)
# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
