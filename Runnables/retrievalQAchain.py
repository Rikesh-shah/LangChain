from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import retrieval_qa
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

# initialize the LLM
llm = GoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.7)

# create RetrievalQA chain
qa_chain = retrieval_qa.from_chain_type(llm = llm, retriever = retriever)

# ask a question
query = "What are the kr key takeways from the document?"
answer = qa_chain.run(query)

print(f"Answer : {answer}")