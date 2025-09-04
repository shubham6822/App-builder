from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from prompt import *
from states import *

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

user_prompt = "create a simple calculator web application"
def planner_agent(state:dict) -> dict:
    user_prompt= state["user_prompt"]
    # Getting planning response
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan": resp}



graph = StateGraph(dict)
graph.add_node("planner",planner_agent)
graph.set_entry_point("planner")

agent = graph.compile()

result = agent.invoke({"user_prompt": user_prompt})
print(result)

