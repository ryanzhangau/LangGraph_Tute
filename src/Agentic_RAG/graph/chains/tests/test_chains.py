from pprint import pprint

from Agentic_RAG.graph.chains.generation import generation_chain
from Agentic_RAG.graph.chains.retrieval_grader import GradeDocuments, grade_chain
from Agentic_RAG.ingestion import chroma_retriever


def test_retrieve_grader_answer_yes() -> None:

    query = "agent memory"
    docs = chroma_retriever.invoke(query)
    doc_text = docs[0].page_content

    res: GradeDocuments = grade_chain.invoke({
        "question": query,
        "document": doc_text
    })

    assert res.binary_score == "yes"



def test_retrieve_grader_answer_no() -> None:

    query = "Other questions"
    docs = chroma_retriever.invoke(query)
    doc_text = docs[0].page_content

    res: GradeDocuments = grade_chain.invoke({
        "question": query,
        "document": doc_text
    })

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = chroma_retriever.invoke(question)
    doc_text = docs[0].page_content
    generation = generation_chain.invoke({
        "context": doc_text,
        "question": question
    })
    pprint(generation)
