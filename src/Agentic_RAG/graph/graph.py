from dotenv import load_dotenv
from langgraph.graph import StateGraph

from Agentic_RAG.graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE
from Agentic_RAG.graph.nodes import retrieve_node, grade_document, web_search, generate
from Agentic_RAG.graph.state import GraphState

load_dotenv()

def director_websearch_generate(state: GraphState):
    if state["web_search"]:
        return WEBSEARCH
    return GENERATE

builder = StateGraph(GraphState)
builder.add_node(RETRIEVE, retrieve_node)
builder.add_node(GRADE_DOCUMENTS, grade_document)
builder.add_node(WEBSEARCH, web_search)
builder.add_node(GENERATE, generate)
builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_conditional_edges(GRADE_DOCUMENTS, director_websearch_generate,[WEBSEARCH, GENERATE])
builder.add_edge(WEBSEARCH, GENERATE)
builder.set_finish_point(GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())
