import ollama
import streamlit as st
import camelot
import pdfplumber
import fitz
import re
import sys
import os
import tempfile
import logging
import shutil
import nltk


from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Streamlit UI
st.set_page_config(
    page_icon="✈️",
    page_title="OneCrew Chatbot",
    initial_sidebar_state="collapsed",
    layout="wide",    
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def get_model_names(models_info: Dict[str, List[Dict[str, Any]]],) -> Tuple[str, ...]:
    logger.info("Get Model")
    modnames = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Model Received: {modnames}")
    return modnames

# Create a vector database with tabular and non-tabular data processing
def create_database(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())  
        temp_file.flush()

        # Extract text data from pdf with pdfplumber
        all_text = ""
        with pdfplumber.open(temp_file.name) as pdf:
            for page_number, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                all_text += page_text if page_text else ""
            logger.info(f"Extracted text from {len(pdf.pages)} pages")

        # Extract tables from PDF with Camelot
        tables = camelot.read_pdf(temp_file.name, pages='all')
        if tables:
            logger.info(f"Found {len(tables)} tables")
            for i, table in enumerate(tables):
                table_df = table.df  
                logger.info(f"Table {i+1}: {table_df}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_text(all_text) 

        embeddings = OllamaEmbeddings()
        vector_db = Chroma.from_documents(docs, embeddings)
        
        logger.info("Vector database created from extracted PDF content")
        return vector_db  
    
# Config the Retriever
def rag_retriever(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"""Processing question: {
                question} using model: {selected_model}""")
    llm = ChatOllama(model=selected_model, temperature=0)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant designed to retrieve the most relevant and accurate information from a large database.
        Your task is to process the user’s question, retrieve precise data from the vector database, and generate an accurate and coherent answer.
        Carefully analyze the user question and identify the key information needed.
        Retrieve the most relevant data from the vector database that aligns with the question.
        Ensure that the data retrieved is factually correct and provides comprehensive coverage of the topic.
        If multiple pieces of information are found, combine them logically to formulate a clear and accurate response.
        Always verify that the response is well-structured, concise, and provides the most correct version of the answer.

        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm, 
        prompt=QUERY_PROMPT
    )

    template = """Answer the question strictly based on the following context, without including any outside knowledge or assumptions:
    {context}
    Question: {question}
    If the context does not provide enough information to answer the question, respond with 'The information provided is insufficient to answer the question.' 
    Do not attempt to generate an answer beyond the given {context}. Ensure your response is concise and directly relevant to the question.
    """


    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(uploaded_file) -> List[Any]:
    logger.info(f"""Extracting all pages as images from file: {
                uploaded_file.name}""")
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pages

def delete_db(db: Optional[Chroma]) -> None:
    logger.info("Deleting vector DB")
    if db is not None:
        db.delete_collection()
        st.session_state.pop("pages", None)
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("db", None)
        st.success("Files were deleted successfully!.")
        logger.info("Database and session cleared")
        st.rerun()
    else:
        st.error("There is no database!.")
        logger.warning("Deletion attepmt failed! There is no database!")


def main() -> None:
    st.subheader("🤖 OneCrew Chatbot", divider="gray", anchor=False)

    models_info = ollama.list()
    available_models = get_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "db" not in st.session_state:
        st.session_state["db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick your model!", available_models
        )

    uploaded_file = col1.file_uploader(
        "Upload a PDF file", type="pdf", accept_multiple_files=False
    )

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        if st.session_state["db"] is None:
            st.session_state["db"] = create_database(uploaded_file)
        pages = extract_all_pages_as_images(uploaded_file)
        st.session_state["pages"] = pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in pages:
                    st.image(page_image, width=zoom_level)

    delete_collection = col1.button("⚠️ Delete collection")

    if delete_collection:
        delete_db(st.session_state["db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["db"] is not None:
                            response = rag_retriever(
                                prompt, st.session_state["db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["db"] is None:
                st.warning("Upload a PDF file to begin chat...")

if __name__ == "__main__":
    main()
