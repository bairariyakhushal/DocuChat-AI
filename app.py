import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory

huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

## set up Streamlit 
st.title("Conversational RAG Chatbot")
st.write("Upload Pdf's and chat with their content")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.8
)


session_id = "chat_session"

if "store" not in st.session_state:
    st.session_state.store={}
    
uploaded_files=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

 ## Process uploaded  PDF's
if uploaded_files :
    documents=[]
    
    for uploaded_file in uploaded_files:
        temppdf='./temp.pdf'
        with open(temppdf,'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
            
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    
    # Split and create embeddings for the documents
    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    splitted_docs=splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorDB=Chroma.from_documents(splitted_docs,embeddings)
    retriever=vectorDB.as_retriever()
    
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "            "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt=ChatPromptTemplate.from_messages([
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    
    system_prompt=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    qa_prompt=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])
    
    document_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,document_chain)
    
    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )    
    
    user_query=st.text_input("Enter your question")
    if user_query :
        response=conversational_rag_chain.invoke(
            {"input":user_query},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        st.write("Assistant : ",response['answer'])
    
    