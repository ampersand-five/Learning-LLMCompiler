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

# This method adds conditional edges. What this means is that only one of the downstream
# edges will be taken, and which one that is depends on the results of the start node.
# This takes two required arguments and one optional argument:
# - start_key: A string representing the name of the start node. This key must have
# already been registered in the graph.
# - condition: A function to call to decide what to do next. The input will be the
# output of the start node. It should return a string that is present in
# 'conditional_edge_mapping' and represents the edge to take.
# - (optional) conditional_edge_mapping: A mapping of string to string. The keys should be strings
# that may be returned by condition. The values should be the downstream node to call if
# that condition is returned.
workflow.add_conditional_edges(
    start_key="joiner",
    # Next, we pass in the function that will determine which node is called next.
    condition=should_continue
)

workflow.set_entry_point("plan_and_schedule")

chain = workflow.compile()




# Example Usage


# # *** Example 1 - Simple Question
# for step in chain.stream([HumanMessage(content="What's the GDP of New York?")]):
#     print(step)
#     print("---")

# # Final answer
# print(step[END][-1].content)


# # *** Example 2 - Multi-Hop Question
# steps = chain.stream(
#     [HumanMessage(content="What's the oldest parrot alive, and how much longer is that than the average?")],
#     {"recursion_limit": 100}
# )
# for step in steps:
#     print(step)
#     print("---")

# # Final answer
# print(step[END][-1].content)


# *** Example 3 - Multi-Step Math Question
# for step in chain.stream(
#     [HumanMessage(content="What's ((3*(4+5)/0.5)+3245) + 8? What's 32/4.23? What's the sum of those two values?")]
# ):
#     print(step)
#     print("---")

# # Final answer
# print(step[END][-1].content)