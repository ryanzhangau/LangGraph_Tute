from langchain_core.documents import Document

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