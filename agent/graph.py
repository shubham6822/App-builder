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
    if resp is None:
        raise ValueError("Planner failed")
    return {"plan": resp}

def architecture_agent(state:dict) -> dict:
    plan =  state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))
    if resp is None:
        raise ValueError("Architecture failed")
    resp.plan = plan
    return {"task_plan": resp}

graph = StateGraph(dict)
graph.add_node("planner",planner_agent)
graph.add_node("architecture",architecture_agent)
graph.add_edge("planner", "architecture")
graph.set_entry_point("planner")

agent = graph.compile()

result = agent.invoke({"user_prompt": user_prompt})
print(result)

