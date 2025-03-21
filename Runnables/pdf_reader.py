from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import google_palm
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# load the document
loader = TextLoader("Runnables\openvpn1111.txt")
documents = loader.load()

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlaps = 50)
docs = text_splitter.split_documents(documents)

# convert the text into embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model = "gemini-1.5-pro"))

# create a retriever (fetches relevant documents)
retriever = vectorstore.as_retriever()

# manually retrieve relevant documents
query = "What are the key takeways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# combine retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# initialize the LLM
llm = GoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.7)

# manually pass retrieved text to the LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.predict(prompt)

# print the answer
print("Answer : ", answer)