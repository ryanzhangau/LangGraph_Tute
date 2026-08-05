from typing import Any, Dict

from Agentic_RAG.graph.chains.retrieval_grader import grade_chain
from Agentic_RAG.graph.state import GraphState


def grade_document(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flat to run web search

    Arguments:
        state (dict) -- The current graph state
    Returns:
         state (dict) -- Filtered out irrelevant documents and update web_search state
    """
    documents = state["documents"]

    filtered_documents = []

    for document in documents:
        score = grade_chain.invoke({
            "question": state["question"],
            "document": document
        })

        if score.binary_score == "yes":
            filtered_documents.append(document)

    return {
        "documents": filtered_documents,
        "question": state["question"],
        "web_search": len(filtered_documents) != len(documents)
    }
