from llama_index.llms.cohere import Cohere
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from dotenv import load_dotenv
import os

# load .env file
load_dotenv()
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# Create the embedding model
embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# load the Cohere command model
cohere_model = Cohere(api_key=COHERE_API_KEY, model="command-r-plus")

# Set llm and embed settings to use the Cohere models
Settings.llm = cohere_model
Settings.embed_model = embed_model

# Load data and build an index
documents = SimpleDirectoryReader("data").load_data()  # load data from the data folder
index = VectorStoreIndex.from_documents(
    documents
)  # build an index from the documents in the data folder

# query the index
query_engine = index.as_query_engine()  # create a query engine from the index
response = query_engine.query(
    "What are the additional costs that a home buyer needs to pay in Ontario?"
)
print(response)
