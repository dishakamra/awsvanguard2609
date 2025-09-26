from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
import boto3
import re
import numexpr as ne

bedrock_client = boto3.client(service_name="bedrock-runtime")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

react_agent_llm = ChatBedrock(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={"temperature": 0.0},
)

# --- ReAct prompt ---
prompt_template = """Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- Use CityDistanceLookup for geographic distances between places.
- Use Calculator ONLY for arithmetic on numbers, and provide a pure expression as input (e.g., "383/40" or "((120+40)/5)").
- Do not send non-numeric or knowledge queries to Calculator.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# --- Distance lookup tool (stub) ---
def _lookup_distance_miles(query: str) -> str:
    """
    Minimal stub: returns a typical driving distance for SF <-> LA.
    Replace with a real search API/tool if desired.
    """
    q = query.lower()
    if "san francisco" in q and "los angeles" in q:
        return "383"  # typical driving miles
    return "ERROR: Unknown route. Provide two city names like 'distance between A and B in miles'."

distance_tool = Tool.from_function(
    func=_lookup_distance_miles,
    name="CityDistanceLookup",
    description="Looks up city-to-city distance in miles. Input like: 'distance between San Francisco and Los Angeles in miles'. Returns ONLY a number.",
)

# --- Calculator tool using numexpr (no LLM; no parsing issues) ---
_NUMERIC_EXPR_RE = re.compile(r"^[0-9\.\s\+\-\*\/\(\)eE]+$")

def _safe_calculator(expr: str) -> str:
    expr = expr.strip()
    if not _NUMERIC_EXPR_RE.fullmatch(expr):
        return "ERROR: Calculator only accepts a numeric expression like '383/40'."
    try:
        val = ne.evaluate(expr)
        # numexpr returns a numpy scalar/array; cast to plain string
        return str(float(val))
    except Exception as e:
        return f"ERROR: Failed to evaluate expression: {e}"

calculator_tool = Tool.from_function(
    func=_safe_calculator,
    name="Calculator",
    description="Arithmetic only. Input MUST be a pure numeric expression (e.g., '383/40'). Returns a numeric string.",
)

tools = [distance_tool, calculator_tool]

# --- Build agent ---
react_agent = create_react_agent(
    react_agent_llm,
    tools,
    PromptTemplate.from_template(prompt_template),
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

# --- Run ---
agent_executor.invoke({
    "input": "What is the distance between San Francisco and Los Angeles? If I travel from San Francisco to Los Angeles with the speed of 40MPH how long will it take to reach?"
})
