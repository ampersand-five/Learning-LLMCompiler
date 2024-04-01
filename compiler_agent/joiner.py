from typing import Any, Union, List
import json
from langchain_core.messages import (
  BaseMessage,
  HumanMessage,
  SystemMessage
)

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import SystemMessagePromptTemplate


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

# Read config file
with open('compiler_agent/config.json', 'r') as f:
  config = json.load(f)

# Read joiner prompt from local file
with open('compiler_agent/prompts/joiner_1.txt', 'r') as file:
  joiner_prompt_1 = file.read()
with open('compiler_agent/prompts/joiner_2.txt', 'r') as file:
  joiner_prompt_2 = file.read()

# Since this one has input variables, it is set separately
joiner_prompt_template_2 = SystemMessagePromptTemplate.from_template(template=joiner_prompt_2)

joiner_prompt = ChatPromptTemplate.from_messages(
  [
    SystemMessage(content=joiner_prompt_1),
    MessagesPlaceholder(variable_name='messages'),
    joiner_prompt_template_2
  ]
)

# Optional: set any examples here
joiner_prompt = joiner_prompt.partial(examples='')

llm = ChatOpenAI(**config['joiner_llm'])

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