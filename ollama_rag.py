#### ollama

# Core imports
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

# 1) Start Qdrant (locally or cloud), then:
client = QdrantClient(url=os.getenv("QDRANT_URL","http://localhost:6333"))

# 1) Global Settings
Settings.llm = Ollama(
    model=os.getenv("OLLAMA_MODEL","mistral:latest"),
    base_url=os.getenv("OLLAMA_BASE_URL","http://localhost"),
    request_timeout=120.0
)
Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
Settings.chunk_size = 512


# 2) Build StorageContext with QdrantVectorStore
storage_ctx = StorageContext.from_defaults(
    vector_store=QdrantVectorStore(
        client=client,
        collection_name="llamaindex_rag",
    )
)

# 2) Load documents
documents = SimpleDirectoryReader("data").load_data()

# 3) Build index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_ctx,
    show_progress=True
)

# 4) Query
query_engine_ollama = index.as_query_engine()
response_ollama = query_engine_ollama.query("What are the common symptoms of diabetes?")


from llama_index.core.response.pprint_utils import pprint_response
pprint_response(response_ollama,show_source=False)
