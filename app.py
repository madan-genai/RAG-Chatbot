import streamlit as st
from rag_pipeline import create_vectorstore, build_rag_chain
import tempfile

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot with Groq + LangGraph")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("Document uploaded successfully!")

    vectorstore = create_vectorstore(tmp_path)
    rag_chain = build_rag_chain(vectorstore)

    question = st.text_input("Ask a question from document")

    if question:
        response = rag_chain.invoke(question)
        st.write("### Answer:")
        st.write(response)