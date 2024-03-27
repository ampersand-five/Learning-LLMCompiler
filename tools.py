import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMMathChain


os.environ.get("TAVILY_API_KEY")

# Math tool
llm_math_chain = LLMMathChain.from_llm(
  llm=ChatOpenAI(model="gpt-4-turbo-preview"),
)

math_tool = Tool(
  name='Math',
  description='Math tool.',
  func=llm_math_chain.run
)

search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)

tools = [search, math_tool]

# math_tool.invoke(
#     {
#         "problem": "What's the temp of sf + 5?",
#         "context": ["Thet empreature of sf is 32 degrees"],
#     }
# )