#### ollama

# Core imports
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 1) Global Settings
Settings.llm = Ollama(
    model="mistral:latest",
    base_url="http://216.48.179.162",
    request_timeout=120.0
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
query_engine_ollama = index.as_query_engine()
response_ollama = query_engine_ollama.query("What are the common symptoms of diabetes?")


#### groq API

# from dotenv import load_dotenv
# load_dotenv()
# import os
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# # Core imports
# from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq

# # 1) Set global LLM to Groq
# Settings.llm = Groq(
#     model="llama3-8b-8192",
# )
# Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
# Settings.chunk_size = 512

# # 2) Load documents
# documents = SimpleDirectoryReader("data").load_data()

# # 3) Build index
# index = VectorStoreIndex.from_documents(
#     documents,
#     show_progress=True
# )

# # 4) Query
# query_engine = index.as_query_engine()
# response_groq = query_engine.query("What are the common symptoms of diabetes?")


from llama_index.core.response.pprint_utils import pprint_response
pprint_response(response_ollama,show_source=False)
# pprint_response(response_groq,show_source=False)


from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

retriver=VectorIndexRetriever(index=index,similarity_top_k=3)
postprocess=SimilarityPostprocessor(similarity_cutoff=0.80)

query_engine = RetrieverQueryEngine(retriever=retriver,node_postprocessors=[postprocess])
response = query_engine.query("What are the common symptoms of diabetes?")
pprint_response(response,show_source=True)




# Set up the embedding model using Hugging Face's Instructor model
# embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")

# Use Groq's ChatGroq model via LangChain
# class GroqLLM:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.groq_client = ChatGroq(model_name="qwen-qwq-32b", api_key=self.api_key)

#     def _call(self, prompt: str) -> str:
#         return self.groq_client.chat(prompt)

#     @property
#     def _identifying_params(self) -> dict:
#         return {"model_name": "qwen-qwq-32b"}

# # Initialize LangChain LLM with Groq
# llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))


# # Load documents from the "data" directory
# documents = SimpleDirectoryReader("data").load_data()

# # Create a service context
# service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)

# # Build the index
# index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context)

# # Query the index
# query_engine = index.as_query_engine()
# response = query_engine.query("What is the main theme of this document?")
# print(response,"%%%%%%%%%%%%%%")

















# from dotenv import load_dotenv
# load_dotenv()
# import os
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
# documents = SimpleDirectoryReader("data").load_data()
# print(documents)

# from llama_index.embeddings.instructor import InstructorEmbedding
# from langchain_groq import ChatGroq
# from llama_index.llms.langchain import LangChainLLM


# embed_model=InstructorEmbedding(model_name="hkunlp/instructor-xl")

# llm = LangChainLLM(
#     llm=ChatGroq(
#         model_name="llama3-8b-8192",
#         api_key=os.getenv("GROQ_API_KEY"),
#     )
# )

# from llama_index.core import ServiceContext
# service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
# index = VectorStoreIndex.from_documents(documents,show_progress=True,service_context=service_context) # First it will convert the pdf text in to embedding then it will index



# from dotenv import load_dotenv
# load_dotenv()

# import os
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.embeddings.instructor import InstructorEmbedding
# from llama_index.llms.langchain import LangChainLLM
# from langchain_groq import ChatGroq
# from langchain.chat_models import ChatOpenAI  # fallback if needed

# # Load documents
# documents = SimpleDirectoryReader("data").load_data()

# # Set up InstructorEmbedding
# embed_model = InstructorEmbedding(model_name="hkunlp/instructor-xl")  # or instructor-large

# # Set up Groq LLM
# llm = LangChainLLM(
#     llm=ChatGroq(
#         model_name="llama3-8b-8192",
#         api_key=os.getenv("GROQ_API_KEY"),
#     )
# )

# # Create a service context using local embed + Groq LLM
# from llama_index.core import ServiceContext
# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
# )

# # Create index
# index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)

# Query
# query_engine = index.as_query_engine()
# response = query_engine.query("What are the common symptoms of diabetes?")
# print(response)
