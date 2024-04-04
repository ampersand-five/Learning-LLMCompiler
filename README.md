# LangChain-Self-RAG-Learning
Following the LangChain LLMCompiler example found at: https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb

From the article at: https://blog.langchain.dev/planning-agents/

I take what's there and follow it. I however pulled the code out into their own file
structure rather than one long script. I have also copied out the prompts and changed
them to work better.

Requires a .env file that has the following:
```
# DO NOT COMMIT TO GIT
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
# Optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT="LLMCompiler Tutorial"
```


There is still a problem with the math function not always working. Here's an example:
```python
# *** Example 2 - Multi-Hop Question
steps = chain.stream(
  [HumanMessage(content="What's the oldest parrot alive, and how much longer is that than the average?")],
  {"recursion_limit": 60}
)
for step in steps:
  print(step)
  print("---")

# Final answer
print(step[END][-1].content)
'''
This example #2 question highlights an issue with the math function that needs to be
fixed somehow. Here's the output it wants to generate:

1. tavily_search_results_json(query="oldest parrot alive")
2. tavily_search_results_json(query="average lifespan of a parrot")
3. math(problem="${1} - ${2}", context=["oldest parrot alive", "average lifespan of a parrot"])
4. join()<END_OF_PLAN>

Step 3 is the problem. It should be like this:
3. math(problem="oldest parrot alive minus average lifespan of a parrot", context=[${1},${2}])

Couple issues or fixes:
- The context parameters will fill in with search results, but they don't say what query
  generated those results, and this might be crucial information.
- We could have the llm just fill the 'problem' param with everything, the query and
  the context.
- We probably need a processing step before the math step, it likely would be part of
  math function, it would just have a two step process. Which it already has, so maybe
  just tweak it to handle this case.
'''

```