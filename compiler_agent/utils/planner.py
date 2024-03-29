from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_core.messages import FunctionMessage, SystemMessage

from utils.output_parser import LLMCompilerPlanParser
from langchain import hub
from langchain_openai import ChatOpenAI
from utils.tools import tools


def create_planner(
  llm: BaseChatModel,
  tools: Sequence[BaseTool],
  base_prompt: ChatPromptTemplate
):
  '''This function creates a planner'''

  # Create a string of the tools and their descriptions
  tool_descriptions = "\n".join(
    f"{i}. {tool.description}\n"
    for i, tool in enumerate(tools)
  )

  # Create the general planner prompt that doesn't have the replan instructions inserted
  # Take the base prompt and add the number of tools and their descriptions
  planner_prompt = base_prompt.partial(
    replan="",
    num_tools=len(tools),
    tool_descriptions=tool_descriptions,
  )

  # Create the replanner prompt that has the replan instructions inserted
  # Take the base prompt and add the number of tools and their descriptions and the replan instructions
  replanner_prompt = base_prompt.partial(
    replan=" - You are given \"Previous Plan\" which is the plan that the previous agent created along with the execution results "
    "(given as Observation) of each plan and a general thought (given as Thought) about the executed results. "
    "You MUST use these information to create the next plan under \"Current Plan\".\n"
    " - When starting the Current Plan, you should start with \"Thought\" that outlines the strategy for the next plan.\n"
    " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
    " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
    num_tools=len(tools),
    tool_descriptions=tool_descriptions,
  )

  def should_replan(state: list):
    '''
  Determine if the planner should replan - checks if the last message is a system
  message. This means that the agent tried to answer the question but ended up with
  no results and has context from the last attempt in a system message for how
  to proceed.

Example 1) On first pass, normal planning, not replanning:
State:
- HumanMessage(content="What's the GDP of New York?")

Example 2) On second pass after the first attempt didn't provide the answer, replan:
State:
- HumanMessage(content="What's the GDP of New York?")
- FunctionMessage(content="[{'url': 'https://fed.newyorkfed.org/series/RGMP4', 'content': 'Graph and download economic data for Total Real Gross Domestic Product for New York, NY (MSA) (RGMP4) from 2017 to 2022 about New York, NY,\\xa0...'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json')
- AIMessage(content="Thought: The search result provides a URL to a page on the New York Federal Reserve's website that likely contains the information on New York GDP from 2017 to 2022, but the actual GDP value is not provided in the snippet. Without the specific GDP value, the user's question cannot be directly answered.")
- SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP value for New York. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP figure. - Begin counting at : 1')
- AIMessage(content="Thought: The search result provides a link to a potentially relevant source but does not directly answer the user's question with a specific GDP value for New York. To provide a direct answer, more specific data or a summary of the content from the provided URL is required.")
- SystemMessage(content="Context from last attempt: To answer the user's question, we need the specific GDP value for New York, NY (MSA). A direct extraction of this value from the provided URL or a summary of its content would be necessary. The current result only indicates the availability of such data without specifying it.")
    '''
    # Context is passed as a system message
    return isinstance(state[-1], SystemMessage)

  # Wrap the messages in a dictionary for state passing
  def wrap_messages(state: list):
    return {"messages": state}

  def wrap_and_get_last_index(state: list):
    next_task = 0
    # Look for the last (or you could say most recent) tool result (FunctionMessages
    # are results from function calls that are passed back) and get the index.
    # Note: state[::-1] means start at the end of the sequence and step backwards
    # until you reach the start. This effectively reverses the sequence.
    for message in state[::-1]:
      if isinstance(message, FunctionMessage):
        # +1 because we will pass this 
        next_task = message.additional_kwargs["idx"] + 1
        break
    # Example to illustrate
    '''
State:
- HumanMessage(content="What's the GDP of New York?")
- FunctionMessage(content="[{'url': 'https://fed.newyork.org/series/R', 'content': 'Graph and download
economic data for Total Real Gross Domestic Product for NY (R) from 2017 to 2022 about New York, NY,\\xa0...'}]",
additional_kwargs={'idx': 0}, name='tavily_search_results_json')
- AIMessage(content="Thought: The search result provides a URL to a page on the NY Federal Reserve's website that
likely contains the information on NY's GDP from 2017 to 2022, but the actual GDP value is not provided in the snippet.
Without the specific GDP value, the user's question cannot be directly answered.")
- SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP
value for NY. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP
figure.')

In this case, the last FunctionMessage has an idx of 0, so next_task will be set as 1.
The last message, that's being accessed in the line below, will change to:
- SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP
value for NY. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP
figure. - Begin counting at : 1')
    '''
    state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
    return {"messages": state}

  return (
    # Branching logic that determines which prompt we use: planner or replanner.
    # Each branch is a tuple of (condition, action). The first condition that
    # returns True will be the branch that is executed. The final branch is not a
    # tuple, and is the default action to take if none of the conditions return
    # True.
    RunnableBranch(
      # How to read this: should_replan, (wrap_and_get_last_index | replanner_prompt)
      # Call should_replan(), if it returns True, then call
      # wrap_and_get_last_index() and use the output to map to the
      # replanner_prompt. The | character is a special LangChain LCEL operator
      # that connects actions.
      (should_replan, wrap_and_get_last_index | replanner_prompt),
      # Default action to take: (wrap_messages | planner_prompt) -> this is one
      # action to take, even though it looks like two. The | character is a
      # special LangChain LCEL operator that connects the two.
      wrap_messages | planner_prompt,
    )
    | llm
    | LLMCompilerPlanParser(tools=tools)
  )


# Example
llm = ChatOpenAI(model="gpt-4-turbo-preview")

prompt = hub.pull("wfh/llm-compiler")
'''
The prompt pulled from the hub creates a is actually a three prompt list (ChatPromptTemplate object):

1 - System Message:

  - input_variables: num_tools, tool_descriptions
  - Template:
"""Given a user query, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {num_tools} types:
{tool_descriptions}
{num_tools}. join(): Collects and combines results from prior actions.

- An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
- join should always be the last action in the plan, and will be called in two scenarios:
  (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
  (b) if the answer cannot be determined in the planning phase before you execute the plans. Guidelines:
- Each action described above contains input/output types and description.
  - You must strictly adhere to the input and output types for each action.
  - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
- Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
- Each action MUST have a unique ID, which is strictly increasing.
- Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
- Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join
- Ensure the plan maximizes parallelizability.
- Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.
- Never introduce new actions other than the ones provided."""

2 - Messages Placeholder:
  - variable_name=messages
  - input_variables=messages

3 - System Message:
  - input_variables: (none)
  - Template:
"""Remember, ONLY respond with the task list in the correct format! E.g.:
idx. tool(arg_name=args)"""
'''

print('Planner prompt (ChatPromptTemplate object):\n', prompt, '\n')

# This is the primary "agent" in our application
planner = create_planner(llm, tools, prompt)


# Example usage
# example_question = "What's the temperature in SF raised to the 3rd power?"

# for task in planner.stream([HumanMessage(content=example_question)]):
  # print(task["tool"], task["args"])
  # print("---")