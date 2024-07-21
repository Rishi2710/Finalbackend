from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("/Users/rishit/Desktop/Intern/Final.txt") as f:
    doc = f.read()

#text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
texts = text_splitter.create_documents([doc])
len(texts)


import os
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')

#embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_document"
)

#vectorstore

db = Chroma.from_documents(texts, doc_embeddings, persist_directory="./chroma_db")