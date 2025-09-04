from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompt import *
from states import *

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

user_prompt = "create a simple calculator web application"

resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))

print(resp)