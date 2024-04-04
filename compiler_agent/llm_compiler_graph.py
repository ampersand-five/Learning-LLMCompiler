from langgraph.graph import MessageGraph, END
from typing import Dict, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from task_fetching_unit import plan_and_schedule
from joiner import joiner
from langchain.globals import set_verbose, set_debug

set_verbose(True)
# set_debug(True)

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
  '''Example State with SystemMessage, i.e. means we replan:
- HumanMessage(content="What's the GDP of New York?")
- FunctionMessage(content="[{'url': 'https://fed.newyorkfed.org/series/RGMP4', 'content': 'Graph and download economic data for Total Real Gross Domestic Product for New York, NY (MSA) (RGMP4) from 2017 to 2022 about New York, NY,\\xa0...'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json')
- AIMessage(content="Thought: The search result provides a URL to a page on the New York Federal Reserve's website that likely contains the information on New York GDP from 2017 to 2022, but the actual GDP value is not provided in the snippet. Without the specific GDP value, the user's question cannot be directly answered.")
- SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP value for New York. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP figure.\nThe index for the next task or tasks you create is: 1')
- AIMessage(content="Thought: The search result provides a link to a potentially relevant source but does not directly answer the user's question with a specific GDP value for New York. To provide a direct answer, more specific data or a summary of the content from the provided URL is required.")
- SystemMessage(content="Context from last attempt: To answer the user's question, we need the specific GDP value for New York, NY (MSA). A direct extraction of this value from the provided URL or a summary of its content would be necessary. The current result only indicates the availability of such data without specifying it.")

Example State with AIMessage, i.e. means we finish:
- HumanMessage(content="What's the GDP of New York?")
- FunctionMessage(content="[{'url': 'https://fed.newyorkfed.org/series/RGMP4', 'content': 'Graph and download economic data for Total Real Gross Domestic Product for New York, NY (MSA) (RGMP4) from 2017 to 2022 about New York, NY,\\xa0...'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json')
- AIMessage(content="Thought: The search result provides a URL to a page on the New York Federal Reserve's website that likely contains the information on New York GDP from 2017 to 2022, but the actual GDP value is not provided in the snippet. Without the specific GDP value, the user's question cannot be directly answered.")
- SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP value for New York. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP figure.\nThe index for the next task or tasks you create is: 1')
- AIMessage(content="Thought: The search result provides a link to a potentially relevant source but does not directly answer the user's question with a specific GDP value for New York. To provide a direct answer, more specific data or a summary of the content from the provided URL is required.")
- SystemMessage(content="Context from last attempt: To answer the user's question, we need the specific GDP value for New York, NY (MSA). A direct extraction of this value from the provided URL or a summary of its content would be necessary. The current result only indicates the availability of such data without specifying it.")
- AIMessage(content="I'm unable to find the exact GDP value for New York from the provided sources. The information mentions the real GDP of New York from 2017 to 2022 but does not specify the numbers. For the most accurate and up-to-date figures, I recommend checking official economic reports or databases such as the U.S. Bureau of Economic Analysis or Statista directly.")
  '''
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

# *** Example 1 - Simple Question
for step in chain.stream(
  [HumanMessage(content="What's the GDP of New York?")],
  {"recursion_limit": 40}
):
  print(step)
  print("---")
# Final answer
print(step[END][-1].content)


# *** Example 2 - Multi-Hop Question
steps = chain.stream(
  [HumanMessage(content="What's the oldest parrot alive, and how much longer is that than the average?")],
  {"recursion_limit": 60}
)
for step in steps:
  print(step)
  print("---")
# Final answer
print(step[END][-1].content)
'''
This example #2 question highlights an issue with the math function that needs to be
fixed somehow. Here's the output it wants to generate:

1. tavily_search_results_json(query="oldest parrot alive")
2. tavily_search_results_json(query="average lifespan of a parrot")
3. math(problem="${1} - ${2}", context=["oldest parrot alive", "average lifespan of a parrot"])
4. join()<END_OF_PLAN>

Step 3 is the problem. It should be like this:
3. math(problem="oldest parrot alive minus average lifespan of a parrot", context=[${1},${2}])

Couple issues or fixes:
- The context parameters will fill in with search results, but they don't say what query
  generated those results, and this might be crucial information.
- We could have the llm just fill the 'problem' param with everything, the query and
  the context.
- We probably need a processing step before the math step, it likely would be part of
  math function, it would just have a two step process. Which it already has, so maybe
  just tweak it to handle this case.
'''


### The next two examples will be the inverse of eachother to test and make sure it can
### correctly do reasoning to answer the question. In one case the answer will follow
### the question so the answer is straightforward, in the other question the answer will
### be inverted from the question which will make sure the LLM can correctly see this
### and give the right answer still. So between these two answers, one should give an
### amount, but the other should not give an amount, but state that it's already higher.

# *** Example 3 - Multi-Hop Question
steps = chain.stream(
  [HumanMessage(content="How much does Microsoft's market cap need to increase to exceed Apple's market cap?")],
  {"recursion_limit": 60}
)
for step in steps:
  print(step)
  print("---")
# Final answer
print(step[END][-1].content)


# *** Example 4 - Multi-Hop Question
steps = chain.stream(
  [HumanMessage(content="How much does Apple's market cap need to increase to exceed Microsoft's market cap?")],
  {"recursion_limit": 60}
)
for step in steps:
  print(step)
  print("---")
# Final answer
print(step[END][-1].content)


# *** Example 5 - Multi-Step Math Question
for step in chain.stream(
  [HumanMessage(content="What's ((3*(4+5)/0.5)+3245) + 8? What's 32/4.23? What's the sum of those two values?")]
):
  print(step)
  print("---")
# Final answer
print(step[END][-1].content)


# *** Example 6 - Conversational test, it should respond simply with a greeting.
for step in chain.stream(
  [HumanMessage(content="Hello robot! 🤖")]
):
  print(step)
  print("---")

# Final answer
print(step[END][-1].content)