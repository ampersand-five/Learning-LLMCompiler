from typing import Any, Union, List

from langchain_core.messages import (
  BaseMessage,
  HumanMessage,
  SystemMessage
)

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage


class FinalResponse(BaseModel):
  """The final response/answer."""

  response: str


class Replan(BaseModel):
  feedback: str = Field(
    description="Analysis of the previous attempts and recommendations on what needs to be fixed."
  )


class JoinOutputs(BaseModel):
  """Decide whether to replan or whether you can return the final response."""

  thought: str = Field(
    description="The chain of thought reasoning for the selected action"
  )
  action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
  examples=""
) # You can optionally add examples

''' Joiner Prompt:
- input_variables: examples, messages

The prompt pulled from the hub creates a is actually a three prompt list (ChatPromptTemplate object):

1 - System Message:
  - input_variables: (none)
  - Template:
"""Solve a question answering task. Here are some guidelines:
 - In the Assistant Scratchpad, you will be given results of a plan you have executed to answer the user's question.
 - Thought needs to reason about the question based on the Observations in 1-2 sentences.
 - Ignore irrelevant action results.
 - If the required information is present, give a concise but complete and helpful answer to the user's question.
 - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:

Thought: <reason about the task results and whether you have sufficient information to answer the question>
Action: <action to take>
Available actions:
 (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
 (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan"""

2 - Messages Placeholder:
  - variable_name=messages
  - input_variables=messages

3 - System Message:
  - input_variables: examples
  - Template:
"""Using the above previous actions, decide whether to replan or finish. If all the
required information is present, you may finish. If you have made many attempts to find
the information without success, admit so and respond with whatever information you have
gathered so the user can work well with you.

{examples}"""
'''

llm = ChatOpenAI(model="gpt-4-turbo-preview")

runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)

def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
  '''This function parses the LLM the output from the joiner prompt. That prompt asks
  the LLM to proved a thought and action. The thought is if there's enough information
  to answer the user's question. The action should be either 'Finish' or 'Replan'.

  The decision object passed in has a thought and an action. This takes
  the thought and makes an AI message from it and adds it to the response message list
  this function returns. It also checks if the action in the 'decision' object, that is
  passed in, is a 'Replan' action object. If it is, then a SystemMessage is created from
  the feedback the LLM gave when it decided to use a replan object (the LLM explains why
  it chose that, this is our feedback we set) and the SystemMessage is appended to the
  return list.

  Returns either a list that is:
  - [AIMessage]: Indicates the action was 'Finish'.
  - [AIMessage, SystemMessage]: Indicates the action was 'Replan'
  '''
  response = [AIMessage(content=f"Thought: {decision.thought}")]
  if isinstance(decision.action, Replan):
    return response + [
      SystemMessage(
        content=f"Context from last attempt: {decision.action.feedback}"
      )
    ]
  else:
    return response + [AIMessage(content=decision.action.response)]


def select_recent_messages(messages: list) -> dict:
  '''This function returns all messages up to the most recent human input. So if there's
  a bunch of system and ai messages, they're all included till we hit the most recent
  human message, then we cutoff. So if there's a back and forth with the human this will
  only keep system/ai messages since then.
  '''
  selected = []
  # Reverse the list
  for msg in messages[::-1]:
    selected.append(msg)
    if isinstance(msg, HumanMessage):
      break
  # Return reversed list
  return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output

# Example usage
# example_question = "What's the temperature in SF raised to the 3rd power?"
# input_messages = [HumanMessage(content=example_question)] + tool_messages
# joiner.invoke(input_messages)