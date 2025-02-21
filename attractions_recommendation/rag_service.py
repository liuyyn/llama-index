from dotenv import load_dotenv
import os
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import PromptTemplate

from outputs import Attraction, Itinerary


load_dotenv()
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = os.environ["QDRANT_PORT"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

PROMPT_TEMPLATE = """
    Given the context about the main attractions in {query_str}, provide a structured list of notable attractions.
    
    Please provide a response in the following format for each attraction:
    {
        "name": "Name of the attraction",
        "location": "Address of the attraction",
        "description": "Description of the attraction"
    }

    Question: What are the main attractions in {query_str}?
    
    Response must be valid JSON without any other leading or trailing strings. 
    """


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

        self.output_parser = PydanticOutputParser(
            output_cls=Itinerary, pydantic_format_tmpl="{schema}"
        )

        self.prompt_template = PromptTemplate(PROMPT_TEMPLATE)

    def get_data(self):
        """ """
        pass

    def query(self, city: str):
        """
        Query the index with the given city to get attractions reccommendation on.

        Args:
            city (str): The city on which the attractions recommendation is about.
        Returns:
            str: List of attractions in the city.

        """
        query_engine = self.index.as_query_engine(
            text_qa_template=self.prompt_template,
            output_parser=self.output_parser,
        )
        response = query_engine.query(city)
        return response
