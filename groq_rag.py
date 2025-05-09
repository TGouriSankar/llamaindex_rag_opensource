#### groq API

from dotenv import load_dotenv
load_dotenv()
import os
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Core imports
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# 1) Set global LLM to Groq
Settings.llm = Groq(
    model="llama3-8b-8192",
)
Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
Settings.chunk_size = 512

# 2) Load documents
documents = SimpleDirectoryReader("data").load_data()

# 3) Build index
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# 4) Query
query_engine = index.as_query_engine()
response_groq = query_engine.query("What are the common symptoms of diabetes?")


from llama_index.core.response.pprint_utils import pprint_response
pprint_response(response_groq,show_source=False)