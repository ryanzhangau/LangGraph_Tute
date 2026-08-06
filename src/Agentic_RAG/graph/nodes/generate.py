from typing import Any, Dict

from Agentic_RAG.graph.chains.generation import generation_chain
from Agentic_RAG.graph.state import GraphState



def generate(state: GraphState) -> Dict[str, Any]:
    question = state['question']
    documents = state['documents']

    res = generation_chain.invoke({
        'question': question,
        'context': documents
    })

    return {
        'question': question,
        'documents': documents,
        'generated': res
    }