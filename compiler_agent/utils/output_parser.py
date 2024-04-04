import ast
import re
from typing import (
  Any,
  Dict,
  Iterator,
  List,
  Optional,
  Sequence,
  Tuple,
  Union
)

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = "<END_OF_PLAN>"


### Helper functions

def _ast_parse(arg: str) -> Any:
  try:
    return ast.literal_eval(arg)
  except:  # noqa
    return arg


def _parse_llm_compiler_action_args(args: str, tool: Union[str, BaseTool]) -> list[Any]:
  """Parse arguments from a string."""

  if args == "":
    return ()

  if isinstance(tool, str):
    return ()

  extracted_args = {}
  tool_key = None
  prev_idx = None

  for key in tool.args.keys():
    # Split if present
    if f"{key}=" in args:
      idx = args.index(f"{key}=")
      if prev_idx is not None:
        extracted_args[tool_key] = _ast_parse(
          args[prev_idx:idx].strip().rstrip(",")
        )

      args = args.split(f"{key}=", 1)[1]
      tool_key = key
      prev_idx = 0

  if prev_idx is not None:
    extracted_args[tool_key] = _ast_parse(
      args[prev_idx:].strip().rstrip(",").rstrip(")")
    )

  return extracted_args


def default_dependency_rule(idx, args: str) -> bool:
  '''Checks to see if the given index is listed as a dependency in the args string.

  Uses regex to find all instances of the ID_PATTERN in the args string. This pulls out
  numbers that are formatted to show dependencies.
  Example regex matching: $1 or ${1} -> 1

  Then, for all matches it finds, it checks if the index (idx) passed in is in the list.

  For example, if the args string is "query=$1, $2, $3" and idx is 2, then this function
  will return True.

  Purpose is for another function to use this to find when the current index is one of
  the dependencies so that it can be removed from a list of dependencies.

  '''
  matches = re.findall(ID_PATTERN, args)
  numbers = [int(match) for match in matches]
  return idx in numbers


def _get_dependencies_from_graph(
  idx: int, tool_name: str, args: Dict[str, Any]
  ) -> dict[str, list[str]]:
  '''Get dependencies from a graph.'''

  # If the tool is 'join', that's a special case. It's an internal tool that was not one
  # of the ones in the tools.py file or passed in the tools argument to the planner.
  if tool_name == "join":
    # depends on the previous step
    return list(range(1, idx))

  # define dependencies based on the dependency rule in default_dependency_rule
  return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]


class Task(TypedDict):
  idx: int
  tool: BaseTool
  args: list
  dependencies: Dict[str, list]
  thought: Optional[str]


def instantiate_task(
  tools: Sequence[BaseTool],
  idx: int,
  tool_name: str,
  args: Union[str, Any],
  thought: Optional[str] = None,
  ) -> Task:
  """Instantiate a task."""
  # If the tool is 'join', that's a special case. It's an internal tool that was not one
  # of the ones in the tools.py file or passed in the tools argument to the planner.
  if tool_name == "join":
    tool = "join"

  # Look for the tool in the tools list
  else:
    try:
      # Get the tool names in a list. Then, get the index of the tool_name we're looking
      # for. Finally, using the index we found, use that index to index into the tools
      # list to get out the tool we're looking for.
      tool = tools[[tool.name for tool in tools].index(tool_name)]

    except ValueError as e:
      raise OutputParserException(f"Tool {tool_name} not found.") from e

  # Parse args and dependencies
  tool_args = _parse_llm_compiler_action_args(args, tool)
  dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)

  return Task(
    idx=idx,
    tool=tool,
    args=tool_args,
    dependencies=dependencies,
    thought=thought
  )


class LLMCompilerPlanParser(BaseTransformOutputParser[dict], extra="allow"):
  """Planning output parser."""

  tools: List[BaseTool]

  def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
    '''processes a task list in the following form:
    1. tool_1(arg1="arg1", arg2=3.5, ...)
    2. tool_2(arg1="", arg2="${1}")'
    3. join()<END_OF_PLAN>"

    The ${#} placeholders are variables. These are used to route tool (task) outputs to other tools.
    '''
    texts = []
    # TODO: Cleanup tuple state tracking here.
    # The above is a note from the original walkthrough, I do not know what they were
    # getting at exactly, but I might have an idea:
    # 1) The parse_task() function doesn't really handle when tasks are malformatted.
    #   It will return a None and that gums up the agent a bit and it flails a little
    #   (read: loops plan/join) and eventually the LLM gives a good task and it
    #   continues on.
    # 2) I have noticed that the parse_task() function returns a tuple: (task, thought)
    #   but thought is always discarded and not used.
    thought = None
    for chunk in input:
      # Assume input is str. TODO: support vision/other formats
      text = chunk if isinstance(chunk, str) else str(chunk.content)
      # Example partial text:
      # - texts: ['', '0', '.', ' tav']
      # - text: 'ily'
      # Example complete text:
      # - texts: ['', '0', '.', ' tav', 'ily', '_search', '_results', '_json', '(query', '="', 'G', 'DP', ' of', ' New', ' York', '")', '']
      # - returned task: {'idx': 0, 'tool': TavilySearchResults(description='tavily_search_results_json(query="the search query") - a search engine.', max_results=1), 'args': {'query': 'GDP of New York'}, 'dependencies': [], 'thought': None}
      # - thought (or '_' as we set it here since we're not using it): None
      # Notes:
      # - ingest_token calls _parse_task internally

      # ingest_token will generally just buffer and return, but if there's a newline,
      # it will check if the current buffer is a task. If it is a task it will yield it,
      # clear the buffer and parsing can continue.
      for task, thought in self.ingest_token(token=text, buffer=texts, thought=thought):
        yield task

    # If parsing is complete, either there was just one task or this is the last task.
    # This then processes the last and/or single task in the buffer(texts is the buffer).
    if texts:
      task, _ = self._parse_task("".join(texts), thought)
      # Example:
      # - texts: ['', '0', '.', ' tav', 'ily', '_search', '_results', '_json', '(query', '="', 'G', 'DP', ' of', ' New', ' York', '")', '']
      # - returned task: {'idx': 0, 'tool': TavilySearchResults(description='tavily_search_results_json(query="the search query") - a search engine.', max_results=1), 'args': {'query': 'GDP of New York'}, 'dependencies': [], 'thought': None}
      # - thought (or '_' as we set it here since we're not using it): None
      if task:
        yield task

  def parse(self, text: str) -> List[Task]:
    ''''''
    return list(self._transform([text]))

  def stream(
    self,
    input: str | BaseMessage,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
  ) -> Iterator[Task]:
    '''This takes streamed tokens from the 'planner' Runnable llm output that is
    will only hit this when '''
    yield from self.transform([input], config, **kwargs)

  def ingest_token(
    self,
    token: str,
    buffer: List[str],
    thought: Optional[str]
  ) -> Iterator[Tuple[Optional[Task], str]]:
    '''Appends token to buffer. Checks if the token is a newline. This indicates that
    the plan had multiple steps and there is one on each line. When there is a newline
    it will check if the current buffer is a task and if it is it will yield the task
    and clear the buffer to remove it and parsing can continue.
    '''

    buffer.append(token)

    # If there's a newline, check if the buffer holds a task, meaning there might be
    # multiple tasks, one on each line. We can yield the task and clear it from the
    # buffer and return to allow parsing to continue.
    if "\n" in token:
      buffer_ = "".join(buffer).split("\n")
      suffix = buffer_[-1]
      for line in buffer_[:-1]:
        task, thought = self._parse_task(line, thought)
        if task:
          yield task, thought
      buffer.clear()
      buffer.append(suffix)

  def _parse_task(self, line: str, thought: Optional[str] = None):
    '''This function is used to parse streamed tokens. If what is passed in is not
    complete then it is ignored and simply returned. If it is complete, then we
    check if the streamed tokens were a thought or an action. We check completeness
    by using regex patterns for a THOUGHT_PATTERN and ACTION_PATTERN
    - If Thought: We set the 'thought' variable and keep task as None. Task can be
      checked for None to see if it was a thought.
    - If Task: We pull out the index, tool_name and args and instantiate a task object.
      We set thought to None and return both: (task, thought).
    - If Neither (Incomplete): We return task=None, thought=thought (thought stays as
      it was when passed in as an argument to this function).
    
    Action example line:
    - '0. tavily_search_results_json(query="GDP of New York")'

    Action pattern examples: 
    - 2. join()
      - Good.
    - 2. join() <END_OF_PLAN>
      - Good.
    - 2. join() # call join to get results back
      - Good. Might have a comment, that's ok we capture it and throw it away.
    - join
      - Bad. Missing index and function parenthesis.
    - join()
      - Bad. Missing index.
      
      When bad patterns come through, they should fall through both thought and action
      regex patterns and a 'None' will be returned and None is not to be added to a
      task list.
    '''

    task = None

    if match := re.match(THOUGHT_PATTERN, line):
      # Optionally, action can be preceded by a thought
      thought = match.group(1)

    elif match := re.match(ACTION_PATTERN, line):
      # if action is parsed, return the task, and clear the buffer
      idx, tool_name, args, _ = match.groups()
      idx = int(idx)

      task = instantiate_task(
        tools=self.tools,
        idx=idx,
        tool_name=tool_name,
        args=args,
        thought=thought,
      )

      thought = None

    # Else it is just dropped
    return task, thought