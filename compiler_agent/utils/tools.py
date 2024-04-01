import json
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from utils.math_tools import get_math_tool


# Read config file
with open('compiler_agent/config.json', 'r') as f:
  config = json.load(f)

calculate = get_math_tool(ChatOpenAI(**config['math_llm']))
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