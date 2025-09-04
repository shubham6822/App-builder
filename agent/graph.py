from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

user_prompt = "create a simple calculator web application"

planner_prompt = f"""
You are the PLANNER agent. Convert the user prompt into a COMPLETE engineering project plan

User request: {user_prompt}
"""


class File(BaseModel):
    path: str = Field(description="The path to the file to be created or modified")
    purpose: str = Field(
        description="The purpose of the file, e.g. 'main application logic', 'data processing module', etc.")


class Plan(BaseModel):
    name: str = Field(description="The name of app to be built")
    description: str = Field(
        description="A one-line description of the app to be built, e.g. 'A web application for managing personal finances'")
    techstack: str = Field(
        description="The tech stack to be used for the app, e.g. 'python', 'javascript', 'react', 'flask', etc.")
    features: list[str] = Field(
        description="A list of features that the app should have, e.g. 'user authentication', 'data visualisation', etc.")
    files: list[File] = Field(description="A list of files to be created, each with a 'path' and 'purpose'")


resp = llm.with_structured_output(Plan).invoke(planner_prompt)

print(resp)