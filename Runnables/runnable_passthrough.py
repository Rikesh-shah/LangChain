from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a joke about {topic}",
    input_variables = ["topic"]
)

model = GoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.7)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Explain the following joke - {text}",
    input_variables = ['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic" : "cricket"}))