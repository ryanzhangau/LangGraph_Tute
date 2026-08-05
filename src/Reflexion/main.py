from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

from tool_exector import execute_tools
from reflexion_chains import first_responder, revisor

MAX_TRY = 2
REVISOR = 'revisor'
DRAFT = 'draft'
TOOL_EXECUTOR = 'execute_tools'

def draft_node(state: MessagesState):
    res = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [res]}

def revisor_node(state: MessagesState):
    res = revisor.invoke({"messages": state["messages"]})
    return {"messages": [res]}

def event_loop(state: MessagesState) -> Literal[REVISOR, END]:
    count = sum(isinstance(item, ToolMessage) for item in state['messages'])
    if count > MAX_TRY:
        return END
    return TOOL_EXECUTOR

builder = StateGraph(MessagesState)
builder.add_node(DRAFT, draft_node)
builder.add_node(TOOL_EXECUTOR, execute_tools)
builder.add_node(REVISOR, revisor_node)
builder.add_edge(START, DRAFT)
builder.add_edge(DRAFT, TOOL_EXECUTOR)
builder.add_edge(TOOL_EXECUTOR, REVISOR)
builder.add_conditional_edges(REVISOR, event_loop, [TOOL_EXECUTOR, END])

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("LangGraph Reflexion")
    res = graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
            }
        ]
    })

    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])