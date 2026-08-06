import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from Agentic_RAG.loaders.WebBaseLoader import native_load_webpage


load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs_list = [native_load_webpage(url) for url in urls]

text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=30
)

docs_spliter = text_spliter.split_documents(docs_list)

vectors = Chroma.from_documents(
    documents=docs_spliter,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="qwen3-embedding:8b"),
    persist_directory="./.chroma"
)

chroma_retriever = vectors.as_retriever()