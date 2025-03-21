from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature= 0.7)
# create a prompt template
prompt = PromptTemplate(
    input_variables= ['topi'],
    template = "Suggest a catchy blog title about {topic}."
)

# define the input
topic = input("Enter a topic")

# format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic = topic)

# call teh LLM directly
blog_title = llm.predict(formatted_prompt)

# print the output
print("Generated Blog Title : ", blog_title)