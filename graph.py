
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from nodes import call_model
from edges import route_model_output
from state import InputState, State
from tools import TOOLS
from configuration import Configuration


def get_ada_graph():
    # Define a new graph
    builder = StateGraph(State, input=InputState, config_schema=Configuration)

    # Define the two nodes we will cycle between
    builder.add_node(call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    # Set the entrypoint as `call_model`
    # This means that this node is the first one called
    builder.add_edge("__start__", "call_model")
    
    # Add a conditional edge to determine the next step after `call_model`
    builder.add_conditional_edges(
        "call_model",
        # After call_model finishes running, the next node(s) are scheduled
        # based on the output from route_model_output
        route_model_output,
    )

    # Add a normal edge from `tools` to `call_model`
    # This creates a cycle: after using tools, we always return to the model
    builder.add_edge("tools", "call_model")

    # Compile the builder into an executable graph
    ada_graph = builder.compile(name="Ada Agent")

    return ada_graph


if __name__ == "__main__":
    # This is just for testing purposes
    graph = get_ada_graph()

    input_state = InputState(messages=[
        # Example initial messages can be added here
        # HumanMessage(content="Hello, how can I help you?"),
    ])

    res = graph.invoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},
        {"configurable": {"system_prompt": "You are a helpful AI assistant."}},
    )
    print(res)