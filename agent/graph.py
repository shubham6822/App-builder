from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

res = llm.invoke("who created python answer in ! sentence")

print(res.content)

