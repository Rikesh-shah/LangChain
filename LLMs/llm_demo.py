# from langchain_openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

# result = llm.invoke("What is the capital of Nepal?")

# print(result)

from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-pro")

result = llm.invoke("What is the capital of Nepal?")

print(result)
