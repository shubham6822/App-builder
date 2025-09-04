from langchain.agents import create_react_agent
from langchain.globals import set_verbose, set_debug
from langgraph.constants import END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from agent.prompt import *
from agent.states import *
from agent.tools import *

load_dotenv()
set_debug(True)
set_verbose(True)

llm = ChatGroq(model="openai/gpt-oss-120b")


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

def coder_agent(state:dict) -> dict:
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    react_agent.invoke({"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]})

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}

graph = StateGraph(dict)

#Node
graph.add_node("planner",planner_agent)
graph.add_node("architecture",architecture_agent)
graph.add_node("coder",coder_agent)

# Edge
graph.set_entry_point("planner")
graph.add_edge("planner", "architecture")
graph.add_edge("architecture", "coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)


agent = graph.compile()

if __name__ == "__main__":
    user_prompt = "create a simple calculator web application"
    result = agent.invoke({"user_prompt": user_prompt})
    print(result)

