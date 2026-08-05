from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings


load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [UnstructuredLoader(web_url=url, chunking_stretegy="basic", max_characters=1000000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

docs_spliter = text_spliter.split_documents(docs_list)
# 
# vectors = Chroma.from_documents(
#     documents=docs_spliter,
#     collection_name="rag-chroma",
#     embedding=OllamaEmbeddings(model="qwen3-embedding:8b"),
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:8b")
).as_retriever()