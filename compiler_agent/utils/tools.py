import json
from langchain_openai import ChatOpenAI

# Tool imports
from langchain_community.tools.tavily_search import TavilySearchResults
from utils.math_tools import get_math_tool


# Read config file
with open('compiler_agent/config.json', 'r') as f:
  config = json.load(f)

calculate = get_math_tool(ChatOpenAI(**config['math_llm']))
search = TavilySearchResults(
  # Setting results to 2 for: shows possible return values from tools, they can be an
  # array, a string, etc. For this tool it's an list of dicts. As a learning/tutorial,
  # this will help illustrate better than 1, that the results are per tool and not
  # standarized. Also not higher than 2 because costs for a free tutorial, keeping it
  # low for that.
  max_results=2,
  # Setting a different description here than the default one.
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