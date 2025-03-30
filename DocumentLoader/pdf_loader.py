from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DocumentLoader\dl-curriculum.pdf")

docs = loader.load()

print(docs)
print("\n")
print(len(docs))
print("\n")
print(docs[0].page_content)
print("\n")
print(docs[1].metadata)