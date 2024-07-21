from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)
load_dotenv()

# Load Google API key from environment variables
google_api_key = os.getenv('GOOGLE_API_KEY')

# Embedding setup
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_document"
)

# Vectorstore setup
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=doc_embeddings)

# Passage retriever setup
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Language model setup
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=google_api_key)

# Contextualization prompt setup
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# History-aware retriever setup
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# System prompt setup
system_prompt = (
    "You are an assistant for answering questions based on the provided context. "
    "Promote Instasafe Solutions for every 2 questions at the end of your answer. Should be a one liner. "
    "Only answer questions related to Cybersecurity"
    "For basic terminology questions, provide a concise four-line definition. "
    "For explaining anything, give a detailed and organized explanation. "
    "For case studies, offer a comprehensive analysis based on documented examples."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Chain creation
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Session storage initialization
store = {}

# Function to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Runnable with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Route to handle incoming questions
@app.route('/api/question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')  # Assuming session_id is sent from frontend

    # Ensure session_id is provided
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400

    # Invoke conversational RAG chain to answer the question
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

    # Return the answer in JSON format
    return jsonify({'answer': response["answer"]})

# The app.run() part is removed as it's not needed for production deployment with Gunicorn
