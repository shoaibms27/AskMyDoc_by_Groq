import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Use HuggingFaceEmbeddings as a fallback
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="AskMyDoc By Shoeb (RAG APP)", page_icon="📄")

# Get Groq API Key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Please set your Groq API Key in the environment variable GROQ_API_KEY.")
    st.stop()

# Initialize components
def initialize_components():
    # Model selection
    model_option = st.sidebar.selectbox(
        "Choose Groq Model",
        ["llama3-70b-8192", "llama3-8b-8192"],
        index=0
    )
    
    # Initialize Groq LLM
    groq_llm = ChatGroq(
        api_key=api_key,
        model_name=model_option,
        temperature=0.1,
        max_tokens=8192
    )
    
    # Initialize Embeddings (Fallback to HuggingFaceEmbeddings)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Use a lightweight model
    )
    
    return groq_llm, embedding_model

def process_pdf(uploaded_file):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        doc.close()
        return pdf_text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def create_vectorstore(text, embeddings):
    """Create FAISS vector store from text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

def main():
    st.title("📄 AskMyDoc By Shoeb (RAG APP)")
    st.markdown("Upload a PDF and ask questions about its content")
    
    # Initialize components
    groq_llm, embedding_model = initialize_components()
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            pdf_text = process_pdf(uploaded_file)
            
            if pdf_text:
                st.success("PDF processed successfully!")
                st.sidebar.info(f"Text length: {len(pdf_text)} characters")
                
                # Create vector store
                with st.spinner("Creating search index..."):
                    vectorstore = create_vectorstore(pdf_text, embedding_model)
                    st.info(f"Created {vectorstore.index.ntotal} text chunks")
                
                # Question input
                question = st.text_input("Ask a question about the document:")
                
                if question:
                    with st.spinner("Generating answer..."):
                        try:
                            # Create and run QA chain
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=groq_llm,
                                chain_type="stuff",
                                retriever=vectorstore.as_retriever(),
                                return_source_documents=False  # Disable returning source documents
                            )
                            response = qa_chain({"query": question})
                            
                            # Display results
                            st.subheader("Answer")
                            st.write(response["result"])
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
    else:
        st.info("Please upload a PDF file to get started.")

if __name__ == "__main__":
    main()
