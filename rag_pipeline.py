import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vectorstore(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def build_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question strictly using the context below.
        If answer is not in context, say "Not found in document".

        Context:
        {context}

        Question:
        {question}
        """
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain