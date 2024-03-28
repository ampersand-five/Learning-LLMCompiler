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
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools),
        tool_descriptions=tool_descriptions,
    )

    # Determine if the planner should replan - checks if the last message is a system message
    def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    # Wrap the messages in a dictionary for state passing
    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        # state[::-1] means start at the end of the sequence and step backwards until you reach the start.
        # This effectively reverses the sequence.
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
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

print('Planner prompt (ChatPromptTemplate object):\n', prompt)

# This is the primary "agent" in our application
planner = create_planner(llm, tools, prompt)


# Example usage
# example_question = "What's the temperature in SF raised to the 3rd power?"

# for task in planner.stream([HumanMessage(content=example_question)]):
#     print(task["tool"], task["args"])
#     print("---")