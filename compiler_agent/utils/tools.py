import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from utils.math_tools import get_math_tool


def _get_pass(var: str):
  if var not in os.environ:
    os.environ[var] = getpass.getpass(f"{var}: ")

# Optional: Debug + trace calls using LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "True"
# os.environ["LANGCHAIN_PROJECT"] = "LLMCompiler"
# _get_pass("LANGCHAIN_API_KEY")
_get_pass("OPENAI_API_KEY")
_get_pass("TAVILY_API_KEY")

calculate = get_math_tool(ChatOpenAI(model="gpt-4-turbo-preview"))
search = TavilySearchResults(
  max_results=1,
  description='tavily_search_results_json(query="the search query") - a search engine.',
)

tools = [search, calculate]

# Test the math tool
# calculate.invoke(
#  {
#    "problem": "What's the temp of sf + 5?",
#    "context": ["Thet empreature of sf is 32 degrees"],
#  }
# )