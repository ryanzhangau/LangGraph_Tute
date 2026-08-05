from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represent the state of the graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: int
    web_search: bool
    documents: List[str]