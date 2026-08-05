from typing import Any, Dict

from Agentic_RAG.graph.state import GraphState
from Agentic_RAG.ingestion import retriever


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}