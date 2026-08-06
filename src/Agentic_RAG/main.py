from dotenv import load_dotenv

from Agentic_RAG.graph.graph import graph

load_dotenv()


if __name__ == "__main__":
    print("Agentic RAG")
    print(graph.invoke(input={"question": "What is windows 11?"}))