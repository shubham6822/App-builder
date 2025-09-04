from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

class Schema(BaseModel):
    price: float
    eps: float

res = llm.with_structured_output(Schema).invoke("Extract Price and EPS from this report:"
"NVIDIA reported quarterly EPS of 2.3 and"
"Their current price is $100.")

print(res)

