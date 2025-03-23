from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template = "write a detailed report on topic {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Summarize the following report - {text}",
    input_variables = ['text']
)

model = GoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.7)

parser = StrOutputParser()

# report_gen_chain = RunnableSequence(prompt1, model, parser)

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic" : "Russia vs ukraine"}))