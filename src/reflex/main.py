from typing import Sequence, List, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from chains import generation_chain, reflection_chain
from langgraph.graph.message import add_messages

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


class MessageGraph(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def generation_node(state: MessageGraph):
    return generation_chain.invoke({"messages": state["messages"]})


def reflection_node(state: MessageGraph) -> MessageGraph:
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())
#
# graph.get_graph().print_ascii()
if __name__ == "__main__":
    inputs = HumanMessage(content="""
    Make this tweet better: "
        @LangChainAI
    - newly Tool Calling feature is seriously underrate.
    After a long wait, it's here-making the  implementation of agents across different models with function calling - super easy..
    Made a video covering their newest blog post
    "
    """)

    response = graph.invoke(inputs)
    print(response)

