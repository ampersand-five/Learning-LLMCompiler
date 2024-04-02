from typing import Sequence
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_core.messages import FunctionMessage, SystemMessage
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate

from utils.output_parser import LLMCompilerPlanParser
from langchain_openai import ChatOpenAI
from utils.tools import tools


# Read config file
with open('compiler_agent/config.json', 'r') as f:
  config = json.load(f)

def create_planner(
  llm: BaseChatModel,
  tools: Sequence[BaseTool],
  base_prompt: ChatPromptTemplate
):
  '''This function creates a planner'''

  # Create a string of the tools and their descriptions
  tool_descriptions = "\n".join(
    f"{i+1}. {tool.description}\n" # +1 to offset the 0 starting index, we want it count normally from 1.
    for i, tool in enumerate(tools)
  )

  # Create the general planner prompt that doesn't have the replan instructions inserted
  # Take the base prompt and add the number of tools and their descriptions
  planner_prompt = base_prompt.partial(
    replan="",
    num_tools=len(tools)+1,# add one because we're adding the join() tool at the end.
    tool_descriptions=tool_descriptions,
  )

  # Read joiner prompt from local file
  with open('compiler_agent/prompts/replan.txt', 'r') as file:
    replan_prompt = file.read()

  # Create the replanner prompt that has the replan instructions inserted
  # Take the base prompt and add the number of tools and their descriptions and the replan instructions
  replanner_prompt = base_prompt.partial(
    replan=replan_prompt,
    num_tools=len(tools)+1,# add one because we're adding the join() tool at the end.
    tool_descriptions=tool_descriptions,
  )

  def should_replan(state: list):
    '''
    Determine if the planner should replan - checks if the last message is a SystemMessage
    object. This means that the agent tried to answer the question but ended up with
    no results and has context from the last attempt in a SystemMessage for how to
    proceed.

    Example 1) On first pass, normal planning, not replanning:
    State:
    - HumanMessage(content="What's the GDP of New York?")

    Example 2) On second pass after the first attempt didn't provide the answer, replan:
    State:
    - HumanMessage(content="What's the GDP of New York?")
    - FunctionMessage(content="[{'url': 'https://fed.newyorkfed.org/series/RGMP4', 'content': 'Graph and download economic data for Total Real Gross Domestic Product for New York, NY (MSA) (RGMP4) from 2017 to 2022 about New York, NY,\\xa0...'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json')
    - AIMessage(content="Thought: The search result provides a URL to a page on the New York Federal Reserve's website that likely contains the information on New York GDP from 2017 to 2022, but the actual GDP value is not provided in the snippet. Without the specific GDP value, the user's question cannot be directly answered.")
    - SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP value for New York. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP figure.\n- The index for the next task or tasks you create is: 1')
    - AIMessage(content="Thought: The search result provides a link to a potentially relevant source but does not directly answer the user's question with a specific GDP value for New York. To provide a direct answer, more specific data or a summary of the content from the provided URL is required.")
    - SystemMessage(content="Context from last attempt: To answer the user's question, we need the specific GDP value for New York, NY (MSA). A direct extraction of this value from the provided URL or a summary of its content would be necessary. The current result only indicates the availability of such data without specifying it.")

    Example 3) Have final answer, finish:
    State:
    - HumanMessage(content="What's the GDP of New York?")
    - FunctionMessage(content="[{'url': 'https://fed.newyorkfed.org/series/RGMP4', 'content': 'Graph and download economic data for Total Real Gross Domestic Product for New York, NY (MSA) (RGMP4) from 2017 to 2022 about New York, NY,\\xa0...'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json')
    - AIMessage(content="Thought: The search result provides a URL to a page on the New York Federal Reserve's website that likely contains the information on New York GDP from 2017 to 2022, but the actual GDP value is not provided in the snippet. Without the specific GDP value, the user's question cannot be directly answered.")
    - SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP value for New York. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP figure.\n- The index for the next task or tasks you create is: 1')
    - AIMessage(content="I was unable to find the specific Gross Domestic Product (GDP) figure for New York. You might want to check the latest statistics on reputable economic or governmental websites for the most current information.")
    '''
    # Context is passed as a system message
    return isinstance(state[-1], SystemMessage)

  # Wrap the messages in a dictionary for state passing
  def wrap_messages(state: list):
    return {"messages": state}

  def wrap_and_get_last_index(state: list):
    next_task_index = 0
    # Look for the last (or you could say most recent) tool result (FunctionMessages
    # are results from function calls that are passed back) and get the index. This
    # gives us the index of the last function called. We can then add 1 to set for the
    # next task we will be creating
    # Note: state[::-1] means start at the end of the sequence and step backwards
    # until you reach the start. This effectively reverses the sequence.
    for message in state[::-1]:
      if isinstance(message, FunctionMessage):
        # +1 because we want the next task to start after the last run function.
        next_task_index = message.additional_kwargs["idx"] + 1
        break
    # Example to illustrate
    '''
    State:
    - HumanMessage(content="What's the GDP of New York?")
    - FunctionMessage(content="[{'url': 'https://fed.newyork.org/series/R', 'content': 'Graph and download
      economic data for Total Real Gross Domestic Product for NY (R) from 2017 to 2022 about New York, NY,\\xa0...'}]",
      additional_kwargs={'idx': 0}, name='tavily_search_results_json')
    - AIMessage(content="Thought: The search result provides a URL to a page on the NY Federal Reserve's website that
      likely contains the information on NY's GDP from 2017 to 2022, but the actual GDP value is not provided in the
      snippet. Without the specific GDP value, the user's question cannot be directly answered.")
    - SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP
      value for NY. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP
      figure.')

    In this case, there was only one FunctionMessage so it has an idx of 0, so next_task_index will be set as 1.
    '''

    '''
    Append to the last message what index the next task needs to have.
    Example:
    The last message, that's being accessed in the line below, will change to (using the example above):
    - SystemMessage(content='Context from last attempt: The information provided does not include the specific GDP
    value for NY. A different source or a direct visit to the provided URL might be necessary to obtain the exact GDP
    figure.
    - From the previous attempt information, thought and context, create a new plan to solve
      the user's query with the utmost parallelizability.
    - You must continue the task index from the end of the previous one. Do not repeat task
      indices.
    - The index to continue from in your new action plan is: 3')
    '''
    state[-1].content = state[-1].content + f'''
- From the previous attempt information, thought and context, create a new plan to solve
  the user's query with the utmost parallelizability.
- You must continue the task index from the end of the previous one. Do not repeat task
  indices.
- The index to continue from in your new action plan is: {next_task_index}'''
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


# Read planner prompt from local file
with open('compiler_agent/prompts/planner_1.txt', 'r') as file:
  planner_prompt_1 = file.read()
with open('compiler_agent/prompts/planner_2.txt', 'r') as file:
  planner_prompt_2 = file.read()

# Since this one has input variables, it is set separately
planner_prompt_template_1 = SystemMessagePromptTemplate.from_template(template=planner_prompt_1)

planner_prompt = ChatPromptTemplate.from_messages(
  [
    planner_prompt_template_1,
    MessagesPlaceholder(variable_name='messages'),
    SystemMessage(content=planner_prompt_2)
  ]
)
# Example
llm = ChatOpenAI(**config['planner_llm'])

# This is the primary "agent" in our application
planner = create_planner(llm, tools, planner_prompt)


# Example usage
# example_question = "What's the temperature in SF raised to the 3rd power?"

# for task in planner.stream([HumanMessage(content=example_question)]):
  # print(task["tool"], task["args"])
  # print("---")