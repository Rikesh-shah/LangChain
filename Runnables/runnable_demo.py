from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

print(model.get_input_jsonschema())
print(model.get_output_jsonschema())