from langgraph.graph import MessageGraph, END
from typing import Dict, List
from langchain_core.messages import BaseMessage, AIMessage
from task_fetching_unit import plan_and_schedule
from joiner import joiner

workflow = MessageGraph()

# Build graph

# Define Nodes
# The name of the node and the function to call when this node is reached. This
# registers and sets them for use.
workflow.add_node(key="plan_and_schedule", action=plan_and_schedule)
workflow.add_node(key="joiner", action=joiner)


# Define edges
# Creates an edge from one node to the next. This means that output of the first node
# will be passed to the next node. It takes two arguments.
# - start_key: A string representing the name of the start node. This key must have
# already been registered in the graph.
# - end_key: A string representing the name of the end node. This key must have already
# been registered in the graph.
workflow.add_edge(start_key="plan_and_schedule", end_key="joiner")

# This condition determines looping logic
def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"


workflow.add_conditional_edges(
    start_key="joiner",
    # Next, we pass in the function that will determine which node is called next.
    condition=should_continue,
)

workflow.set_entry_point("plan_and_schedule")

chain = workflow.compile()