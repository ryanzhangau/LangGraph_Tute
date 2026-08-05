from typing import Any, Dict
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from Agentic_RAG.graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_result=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]

    search_results = web_search_tool.invoke({"query": question})

    joined_results = "\n".join([
        search_result["content"] for search_result in search_results["results"]
    ])

    if documents is not None:
        documents.append(joined_results)
    else:
        documents = [joined_results]

    return {
        "question": question,
        "documents": documents
    }

if __name__ == "__main__":
    res = web_search({"documents": [], "question": "how is the weather like in Melbourne today", "generation": 0, "web_search": True})