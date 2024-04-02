from typing import (
  Any,
  Union,
  Iterable,
  List,
  Dict
)
import traceback
import itertools
from typing_extensions import TypedDict
import time
from concurrent.futures import ThreadPoolExecutor, wait

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.runnables import chain as as_runnable

from utils.output_parser import Task
from utils.planner import planner

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
  # Get all previous tool responses
  results = {}
  # Invert the list
  # For each message, if it is a tool message, add it to the results
  for message in messages[::-1]:
    if isinstance(message, FunctionMessage):
      results[int(message.additional_kwargs["idx"])] = message.content
  # Return all instances of tool responses.
  return results


class SchedulerInput(TypedDict):
  messages: List[BaseMessage]
  tasks: Iterable[Task]


def _execute_task(task, observations, config):
  '''Execute a task with the given observations.'''

  tool_to_use = task["tool"]

  if isinstance(tool_to_use, str):
    return tool_to_use

  args = task["args"]

  try:

    if isinstance(args, str):
      resolved_args = _resolve_arg(args, observations)

    elif isinstance(args, dict):
      resolved_args = {
        key: _resolve_arg(val, observations) for key, val in args.items()
      }

    else:
      # If we reach here, the args have not resolved correctly but also didn't throw an exception. This is an attempt to assign them anyway for the tool being called and see if it works. Since each tool is different, it might work, but will likely fail.
      resolved_args = args

  except Exception as e:
    return (
      f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
      f" Args could not be resolved. Error: {repr(e)}"
    )

  try:
    return tool_to_use.invoke(resolved_args, config)

  except Exception as e:

    return (
      f"ERROR(Failed to call {tool_to_use.name} with args {args}."
      + f" Args resolved to {resolved_args}. Error: {repr(e)})"
    )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):

  if isinstance(arg, str) and arg.startswith("$"):

    try:
      stripped = arg[1:].replace(".output", "").strip("{}")
      idx = int(stripped)

    except Exception:
      return str(arg)

    return str(observations[idx])

  elif isinstance(arg, list):
    return [_resolve_arg(a, observations) for a in arg]

  else:
    return str(arg)


@as_runnable
def schedule_task(task_inputs, config) -> None:
  '''Schedule and run the task and update with results.

  task_inputs is a dictionary and a mutable object in python, so it is being passed by
  reference; this function will modify the object directly through the reference. So the
  caller function will see results in the same object it passed in and this function
  thus won't return any values.

  The passed in observations will have a new entry inserted using the index of the task
  as the key and the results of running the task as the value.
  '''

  task: Task = task_inputs["task"]
  observations: Dict[int, Any] = task_inputs["observations"]

  try:
    observation = _execute_task(task, observations, config)

  except Exception as e:
    # This is an attempt to have the LLM Self correct. If something happens, the
    # exception message is set as the observation and the LLM can decide what to do with
    # the error.
    observation = e

  # Mutable parameter passed by reference, so we don't need to return anything.
  observations[task["idx"]] = observation


def schedule_pending_task(
  task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):

  while True:
    deps = task["dependencies"]

    if deps and (any([dep not in observations for dep in deps])):
      # Dependencies not yet satisfied
      time.sleep(retry_after)
      continue

    schedule_task.invoke({"task": task, "observations": observations})
    break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
  """Group the tasks into a DAG schedule.
  For streaming, we are making a few simplifying assumptions:
  1. The LLM does not create cyclic dependencies.
  2. That the LLM will not generate tasks with future dependencies.
  If this ceases to be a good assumption, you can either
  adjust to do a proper topological sort (not-stream)
  or use a more complicated data structure
  """

  tasks = scheduler_input["tasks"]
  messages = scheduler_input["messages"]

  # If we are re-planning, we may have calls that depend on previous
  # plans. Start with those.
  observations = _get_observations(messages)
  task_names = {}
  args_for_tasks = {}
  originals = set(observations)
  # ^^ We assume each task inserts a different key above to avoid race conditions.

  # List for tasks that have dependencies on other steps, they'll be scheduled tasks.
  futures = []
  retry_after = 0.25  # Retry every quarter second

  with ThreadPoolExecutor() as executor:
    for task in tasks:
      # Grab dependencies
      deps = task["dependencies"]
      # Grab task names and args because outside the thread pool executor, we lose each
      # task except the last one to finish. Grabbing here, let's us keep this info after
      # they all complete.
      task_names[task["idx"]] = (
        task["tool"] if isinstance(task["tool"], str) else task["tool"].name
      )
      args_for_tasks[task["idx"]] = (task["args"])

      if (
        # Depends on other tasks
        deps
        and (any([dep not in observations for dep in deps]))
      ):
        futures.append(
          executor.submit(
            schedule_pending_task, task, observations, retry_after
          )
        )

      else:
        # No deps or all deps satisfied, can schedule now.
        schedule_task.invoke(input={"task": task, "observations": observations})

    # All tasks have been submitted or enqueued. Wait for them to complete.
    wait(futures)

  # Convert observations to new tool messages to add to the state
  new_observations = {
    k: (task_names[k], args_for_tasks[k], observations[k])
    for k in sorted(observations.keys() - originals)
  }

  tool_messages = [
    FunctionMessage(name=task_name, content=str(observation), additional_kwargs={'idx': k, 'args':task_args})
    for k, (task_name, task_args, observation) in new_observations.items()
  ]

  return tool_messages

@as_runnable
def plan_and_schedule(messages: List[BaseMessage], config):

  # Planner is a LangChain Runnable, in this case a chain that plans and returns tasks:
  # return (
  #   RunnableBranch(
  #     (should_replan, wrap_and_get_last_index | replanner_prompt),
  #     wrap_messages | planner_prompt,
  #   )
  #   | llm
  #   | LLMCompilerPlanParser(tools=tools)
  # )
  # Planner returns a generator of tasks, meaning it's a lazy iterator. We will call
  # next() on it to kickstart the first task.
  tasks = planner.stream(messages, config)

  # Get the first task which makes the lazy generator kickstart the first task, then
  # join it back now that the first task has started.
  try:
    first_task = next(tasks)
    tasks = itertools.chain([first_task], tasks)

  except StopIteration:
    # Handle the case where 'tasks' is empty or has reached its end
    tasks = iter([])

  scheduled_tasks = schedule_tasks.invoke(
    {
      "messages": messages,
      "tasks": tasks,
    },
    config,
  )

  return scheduled_tasks


# Example:
# example_question = "What's the temperature in SF raised to the 3rd power?"
# tool_messages = plan_and_schedule.invoke([HumanMessage(content=example_question)])
# print(tool_messages)