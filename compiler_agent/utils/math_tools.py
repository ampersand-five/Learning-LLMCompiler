import math
import re
from typing import List, Optional

import numexpr
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool


_MATH_DESCRIPTION = '''
Solves the provided math problem.
- `problem` can be either a simple math problem (e.g. "1 + 3") or a word problem (e.g.
  "how many apples are there if there are 3 apples and 2 apples").
- You cannot calculate multiple expressions in one call. For instance,
  `math("1 + 3, 2 + 4")` does not work. If you need to calculate multiple expressions,
  you need to call them separately like `math("1 + 3")` and then `math("2 + 4")`
- Minimize the number of `math` actions as much as possible. For instance, instead of
  calling:
  ```
  2. math("what is 10% of ${{1}}")
  3. math("${{1}} + ${{2}}")
  ```
  You MUST instead call:
  ```
  2. math("what is 110% of ${{1}}")
  ```
  Which will reduce the number of math actions.

Optional context:
- You can optionally provide a list of strings as `context` to help the agent solve the
  problem. If there are multiple contexts you need to answer the question, you can
  provide them as a list of strings.
- `math` action will not see the output of the previous actions unless you provide it as
  `context`. You MUST provide the output of the previous actions as `context` if you
  need to do math on it.
- You MUST NEVER provide `tavily_search_results_json` type action's outputs as a
  variable in the `problem` argument. This is because `tavily_search_results_json`
  returns a text blob that contains the information about the entity, not a number or
  value. Therefore, when you need to provide an output of `tavily_search_results_json`
  action, you MUST provide it as a `context` argument to `math` action.
  For example:
  ```
  1. tavily_search_results_json("Barack Obama")
  2. math("age of ${{1}}/2")
  ```
  This NEVER allowed.
  Instead do:
  ```
  1. tavily_search_results_json("Barack Obama")
  2. math("age of Barack Obama divided by two", context=[${{1}}])
  ```
- When you ask a question about `context`, specify the units. For instance, "What is x
  in height?" or "What is x in millions?" instead of "What is x?"
'''

_SYSTEM_PROMPT = '''Translate a math problem into a expression that can be executed
using Python's numexpr library. Use the output of running this code to answer the
question.

Question: ${{Question with math problem.}}
```text
${{single line mathematical expression that solves the problem}}
```
...numexpr.evaluate(text)...
```output
${{Output of running the code}}
```
Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?
ExecuteCode({{code: "37593 * 67"}})
...numexpr.evaluate("37593 * 67")...
```output
2518731
```
Answer: 2518731

Question: 37593^(1/5)
ExecuteCode({{code: "37593**(1/5)"}})
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718
```
Answer: 8.222831614237718
'''

_ADDITIONAL_CONTEXT_PROMPT = '''The following additional context is provided from other
functions. Use it to substitute into any variables or other words in the problem.

Context:
{context}

Note that context varibles are not defined in code yet. You must extract the relevant
numbers and directly put them in code.'''


class ExecuteCode(BaseModel):
  '''The input to the numexpr.evaluate() function.'''

  reasoning: str = Field(
    ...,
    description="The reasoning behind the code expression, including how context is included, if applicable.",
  )

  code: str = Field(
    ...,
    description="The simple code expresssion to execute by numexpr.evaluate().",
  )


def _evaluate_expression(expression: str) -> str:
  try:
    local_dict = {"pi": math.pi, "e": math.e}
    output = str(
      numexpr.evaluate(
        expression.strip(),
        global_dict={},  # restrict access to globals
        local_dict=local_dict,  # add common mathematical functions
      )
    )
  except Exception as e:
    raise ValueError(
      f'Failed to evaluate "{expression}". Raised error: {repr(e)}.'
      " Please try again with a valid numerical expression"
    )

  # Remove any leading and trailing brackets from the output
  return re.sub(r"^\[|\]$", "", output)


def get_math_tool(llm: ChatOpenAI):
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", _SYSTEM_PROMPT),
      ("user", "{problem}"),
      MessagesPlaceholder(variable_name="context", optional=True),
    ]
  )
  extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)

  def calculate_expression(
    problem: str,
    context: Optional[List[str] | str] = None,
    config: Optional[RunnableConfig] = None,
  ):
    chain_input = {"problem": problem}
    # If there's context append it to the prompt.
    if context:
      # Check if the context is string or list, if string leave, if list, make string.
      if isinstance(context, list):
        context_str = "\n".join(context)
      elif isinstance(context, str):
        context_str = context

      # If the context string is not empty after stripping it of whitespace
      if context_str.strip():
        context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
          context=context_str.strip()
        )
        chain_input["context"] = [SystemMessage(content=context_str)]
    code_model = extractor.invoke(chain_input, config)
    try:
      return _evaluate_expression(code_model.code)
    except Exception as e:
      return repr(e)

  return StructuredTool.from_function(
    name="math",
    func=calculate_expression,
    description=_MATH_DESCRIPTION,
  )