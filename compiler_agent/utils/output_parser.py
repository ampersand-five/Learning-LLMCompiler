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
  numbers taht are formatted to show dependencies.
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
    return list(range(1, idx))

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
    thought=thought,
  )


class LLMCompilerPlanParser(BaseTransformOutputParser[dict], extra="allow"):
  """Planning output parser."""

  tools: List[BaseTool]

  def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
    texts = []
    # TODO: Cleanup tuple state tracking here.
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
      for task, thought in self.ingest_token(text, texts, thought):
        yield task
    # Final possible task
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
    ''''''
    yield from self.transform([input], config, **kwargs)

  def ingest_token(
    self,
    token: str,
    buffer: List[str],
    thought: Optional[str]
  ) -> Iterator[Tuple[Optional[Task], str]]:
    '''
    '''
    buffer.append(token)
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
    """This function is used to parse streamed tokens. If what is passed in is not
    complete then it is ignored and simply returned. If it is complete, then we
    check if the streamed tokens were a thought or an action.
    - If Thought: We set the 'thought' variable and keep task as None. Task can be
      checked for None to see if it was a thought.
    - If Task: We pull out the index, tool_name and args and instantiate a task object.
      We set thought to None and return both: (task, thought).
    - If Neither (Incomplete): We return task=None, thought=thought (thought stays as
      it was when passed in as an argument to this function).
    
    Example line:
    - '0. tavily_search_results_json(query="GDP of New York")'
    """
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