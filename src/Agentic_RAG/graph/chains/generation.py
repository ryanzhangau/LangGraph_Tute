from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5",
#     temperature=0
# )

prompt = ChatPromptTemplate.from_messages([
    ("human",
     """You are an assistant for question-answering tasks.
 Use the following peieces of retrieved context to answer the question. 
 If you don't know the answer, just say that you don't know.
 Use three sentences maximum and keep the answer concise.
 
 Question: {question} 
 Context:{context} 
 Answer:""")
])

generation_chain = prompt | llm | StrOutputParser()