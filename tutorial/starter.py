from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

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
