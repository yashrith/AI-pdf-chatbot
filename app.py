import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_pdf(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_quest):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs= new_db.similarity_search(user_quest)

    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs, "question": user_quest},
        return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])
# ---- Streamlit Page Configuration ----
st.set_page_config(page_title="Chat with PDF", layout="wide")

# ---- UI Enhancements ----
st.markdown(
    """
    <h1 style="text-align: center; color: #4A90E2;">📜 Chat with PDF using Gemini 🤖</h1>
    <p style="text-align: center; font-size: 16px;">Upload legal PDFs and ask AI-powered questions!</p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar: Upload PDFs ----
st.sidebar.header("📂 Upload & Process PDFs")
pdf_docs = st.sidebar.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("📥 Submit & Process"):
    if pdf_docs:
        with st.spinner("Processing PDFs... ⏳"):
            extracted_text = get_pdf(pdf_docs)
            text_chunks = get_text_chunks(extracted_text)
            get_vector_store(text_chunks)
            st.sidebar.success("✅ PDFs Processed Successfully!")
    else:
        st.sidebar.error("⚠️ Please upload at least one PDF.")

# ---- Chat Interface ----
st.markdown("<h3 style='text-align: center;'>💬 Ask Your Legal Questions</h3>", unsafe_allow_html=True)

# Chat-style user input
user_question = st.chat_input("Type your question here...")

if user_question:
    with st.spinner("Fetching legal insights..."):
        response = user_input(user_question)

    # Display response in chat-style format


# ---- Footer ----
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 14px; color: gray;">
        Built with ❤️ using Streamlit and Google Gemini AI
    </p>
    """,
    unsafe_allow_html=True,
)
