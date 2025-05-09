from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

# Initialize Qdrant client and vector store
client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="llamaindex_rag")

# 1) Global Settings
Settings.llm = Ollama(
    model="mistral:latest",
    base_url="http://216.48.179.162",
    request_timeout=120.0
)
Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
Settings.chunk_size = 512

# Create a VectorStoreIndex from the vector store
index = VectorStoreIndex.from_vector_store(vector_store)

# Initialize the retriever with the index
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
postprocess=SimilarityPostprocessor(similarity_cutoff=0.80)

# Create the query engine
query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors=[postprocess])

# Query the engine
response = query_engine.query("What are the common symptoms of diabetes?")
pprint_response(response)


# # retrieve_script.py
# # This script retrieves answers from Qdrant-backed LlamaIndex using a low-level retriever API.
# # Ensure you have the necessary dependencies installed:
# # pip install qdrant-client llama-index-vector-stores-qdrant llama-index

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # 1) Load environment variables\load_dotenv()

# # 2) Import Qdrant and LlamaIndex retriever components
# from qdrant_client import QdrantClient
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.indices.postprocessor import SimilarityPostprocessor
# from llama_index.core.response.pprint_utils import pprint_response
# from llama_index.core import Settings

# # 3) Configure your LLM in Settings (uncomment one)
# # Example with Ollama:
# from llama_index.llms.ollama import Ollama
# Settings.llm = Ollama(
#     model="mistral:latest",
#     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#     request_timeout=120.0
# )
# # Example with Groq:
# # from llama_index.llms.groq import Groq
# # Settings.llm = Groq(
# #     model="llama3-8b-8192",
# #     api_key=os.getenv("GROQ_API_KEY")
# # )

# # 4) Initialize Qdrant client and vector store
# qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
# qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
# client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="llamaindex_rag"
# )

# # 5) Create a VectorIndexRetriever for pure ANN search
# retriever = VectorIndexRetriever(
#     index=vector_store,
#     similarity_top_k=3
# )
# # Optional: post-process to filter nodes by similarity score
# postprocessor = SimilarityPostprocessor(similarity_cutoff=0.8)

# # 6) Build a RetrieverQueryEngine using the retriever
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     node_postprocessors=[postprocessor]
# )

# # 7) Define helper function to ask questions and display sources
# def ask(question: str):
#     response = query_engine.query(question)
#     print(f"\nQuestion: {question}\n")
#     print(f"Answer:\n{response.response}\n")
#     print("Source Nodes:")
#     # Pretty-print response with sources
#     pprint_response(response, show_source=True)

# # 8) CLI entry point
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         q = " ".join(sys.argv[1:])
#     else:
#         q = input("Enter your question: ")
#     ask(q)




# # ingest_script.py
# # This script ingests documents into Qdrant using LlamaIndex.
# # Ensure you have `qdrant-client` and `llama-index-vector-stores-qdrant` installed:
# # pip install qdrant-client llama-index-vector-stores-qdrant

# import os
# from dotenv import load_dotenv

# # LlamaIndex core and integrations
# from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient

# # 1) Load environment variables
# load_dotenv()

# # 2) Configure global settings for embeddings (LLM not needed for ingestion)
# Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
# Settings.chunk_size = 512

# # 3) Initialize Qdrant client and vector store
# qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
# qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
# client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
# vector_store = QdrantVectorStore(client=client, collection_name="llamaindex_rag_ollama")

# # 4) Build a StorageContext using the vector store
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# # 5) Load documents and ingest into Qdrant
# # The directory 'data' should contain your PDF or text files
# documents = SimpleDirectoryReader("data").load_data()

# # 6) Create and persist index into Qdrant (embeddings done automatically)
# VectorStoreIndex.from_documents(
#     documents,
#     storage_context=storage_context,
#     show_progress=True
# )

# print("Ingestion completed. Documents indexed into Qdrant collection 'llamaindex_rag_ollama'.")

# # retrieve_script.py
# # This script retrieves answers from Qdrant-backed LlamaIndex using the high-level index API.
# # Ensure you have the same dependencies installed:
# # pip install qdrant-client llama-index-vector-stores-qdrant

# import os
# from dotenv import load_dotenv

# # 1) Load environment variables
# load_dotenv()

# # 2) Initialize Qdrant client and vector store
# from qdrant_client import QdrantClient
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.core import Settings, StorageContext, load_index_from_storage

# qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
# qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
# client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
# vector_store = QdrantVectorStore(client=client, collection_name="llamaindex_rag_ollama")

# # 3) Create a StorageContext
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# # 4) Load the existing index from storage
# index = load_index_from_storage(storage_context)

# # 5) Create a query engine (default uses Settings.llm)
# query_engine = index.as_query_engine()

# # 6) Ask a question
# question = os.getenv("QUERY", "What are the common symptoms of diabetes?")
# response = query_engine.query(question)

# # 7) Pretty-print the response and sources
# from llama_index.core.response.pprint_utils import pprint_response
# print("Response:")
# pprint_response(response, show_source=True)








# # LlamaIndex core and integrations
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# import os

# # 3) Initialize Qdrant client and vector store
# qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
# qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
# client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="llamaindex_rag",
# )

# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.indices.postprocessor import SimilarityPostprocessor
# from llama_index.core.response.pprint_utils import pprint_response

# retriver=VectorIndexRetriever(index=vector_store,similarity_top_k=3)
# postprocess=SimilarityPostprocessor(similarity_cutoff=0.80)

# query_engine = RetrieverQueryEngine(retriever=retriver,node_postprocessors=[postprocess])
# response = query_engine.query("What are the common symptoms of diabetes?")
# pprint_response(response,show_source=True)