import os
import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
from gtts import gTTS
from pinecone import Pinecone, ServerlessSpec
import re
# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant 💬",
    page_icon="📄",
    layout="wide"
)


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.markdown("""
<style>
.main {
    background-color: #F5F5F5;
    padding: 2rem;
}
.stTextInput > div > div > input {
    background-color: white;
    border: 2px solid #4A90E2;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
}
.sidebar .sidebar-content {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clean_text(text):
    # Remove any HTML tags using regex
    clean_text = re.sub(r"<[^>]*>", "", text)
    return clean_text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += clean_text(page_text)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


# Index name
INDEX_NAME = "pdf-chat-index"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))



def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if the index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        # Create the index only if it doesn't exist
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        st.info(f"Created a new Pinecone index: {INDEX_NAME}")
    else:
        st.info(f"Using existing Pinecone index: {INDEX_NAME}")

    # Load the index
    index = pc.Index(INDEX_NAME)

    # Upsert the embeddings into Pinecone
    vectors = [
    {"id": str(i), "values": embeddings.embed_query(chunk), "metadata": {"text": chunk}}
    for i, chunk in enumerate(text_chunks)
    ]
    index.upsert(vectors)
    st.success("Vectors upserted into Pinecone successfully!")



def get_conversational_chain():
    prompt_template = """
    Use the chat history and the provided context to answer the question as detailed as possible.
    If the answer is not in the provided context, just say "Sorry, I didn't understand your question. Do you want to connect with a live agent?".

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {question}

    Answer:
    """
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", generation_config=generation_config)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"],
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


from langchain.schema import Document

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    # Embed query and fetch matching documents
    query_vector = embeddings.embed_query(user_question)
    search_results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    # Convert search results to Document objects
    docs = [
        Document(page_content=res.metadata.get("text", ""), metadata=res.metadata)
        for res in search_results.matches
        if res.metadata and "text" in res.metadata
    ]

    # Prepare chat history
    chat_history_str = "\n".join(
        [f"User: {entry['user']}\nAI: {entry['ai']}" for entry in st.session_state.chat_history[-5:]]
    )

    # Load chain and generate response
    chain = get_conversational_chain()
    response = chain(
        {
            "input_documents": docs,
            "question": user_question,
            "chat_history": chat_history_str,
        },
        return_only_outputs=True,
    )

    # Add AI response to chat history
    ai_response = response.get("output_text", "Sorry, I couldn't process that.")
    context_source = docs[0].page_content if docs else "No context available."
    st.session_state.chat_history.insert(0, {"user": user_question, "ai": ai_response, "context": context_source})


def query_tts(text):
    
    tts = gTTS(text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)   
    return audio_buffer

def play_audio(text):
    audio_buffer = query_tts(text)
    st.audio(audio_buffer, format="audio/mp3")

import html
import streamlit as st

# Function to sanitize HTML input (if needed)
def sanitize_html(text):
    return html.escape(text)

def display_chat_history():
    for idx, entry in enumerate(reversed(st.session_state.chat_history)):
        user_input = sanitize_html(entry["user"])
        ai_response = sanitize_html(entry["ai"])

        st.markdown(
            f"""
            <div style="
                display: flex; 
                justify-content: flex-end; 
                margin-bottom: 10px;
            ">
                <div style="
                    background-color: #E6F3FF; 
                    color: #333; 
                    padding: 10px; 
                    border-radius: 10px; 
                    max-width: 70%; 
                    text-align: right;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                ">
                    {user_input}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
                <div style="
                    background-color: #F0F0F0; 
                    color: #333; 
                    padding: 10px; 
                    border-radius: 10px; 
                    max-width: 70%; 
                    text-align: left;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                ">
                    {ai_response}
                </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(f"🔊 Play Answer {idx + 1}", key=f"play_{idx}"):
            play_audio(entry["ai"])

        with st.expander("Show Context Source", expanded=False):
            st.write(entry["context"])

def main():
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #4A90E2, #50C878);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h1>PDF Chat Assistant 📄💬</h1>
        <p>Upload PDFs and chat with your documents!</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Document Controls")
        
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            accept_multiple_files=True,
            type=['pdf'],
            help="Select one or more PDF files to analyze"
        )

        if pdf_docs:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

        st.markdown("---")
        if st.button("Clear Conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    user_question = st.chat_input("Ask a question about your PDF...")
    
    if user_question:
        user_input(user_question)

    display_chat_history()

if __name__ == "__main__":
    main()
