from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings

import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "OPENAI_API_KEY not found in environment variables."
openai.api_key = openai_api_key

DATA_PATH = "./chanel"

#def main():
# generate_data_store()   

#def generate_data_store():
#    documents = load_documents()
    
    

#def load_documents():
#    loader = DirectoryLoader(DATA_PATH, glob="*.md")
#    documents = loader.load()
#    return documents

#def split_text(documents: list[Document]):
#    text_splitter = RecursiveCharacterTextSplitter(
#        chunk_size=300,
#        chunk_overlap=100,
#        separators=["\n"],
#    )
#    chunks = text_splitter.split_documents(documents)
#    embeddings = OpenAIEmbeddings()
#    vectorstore = FAISS.from_documents(chunks, embeddings)
#    vectorstore.save_local("./")

def retrieval_chain():
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain_core.vectorstores import VectorStoreRetriever

    from langchain_openai import OpenAI
    embeddings = OpenAIEmbeddings()
    retriever = VectorStoreRetriever(vectorstore=FAISS.load_local("./", embeddings, allow_dangerous_deserialization=True))
    combine_docs_chain = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)
    
    query = "who is messi"
    res = combine_docs_chain.invoke({"query": query})
    print(res)
    

retrieval_chain()




