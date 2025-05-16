import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os
import shutil
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from datetime import datetime, timezone
import gc
import time

load_dotenv()

st.set_page_config(page_title="Конституционный Ассистент РК v3", page_icon="🏛️")
st.title("🏛️ Ассистент по Конституции РК (Ollama, ChromaDB, MongoDB)")

OLLAMA_BASE_URL_ENV = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL_ENV = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL_ENV = os.getenv("OLLAMA_CHAT_MODEL", "llama2")
MONGO_URI_ENV = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
CHROMA_PERSIST_DIR_ENV = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_constitution_v3")

OLLAMA_BASE_URL = st.sidebar.text_input("Адрес сервера Ollama", OLLAMA_BASE_URL_ENV)
OLLAMA_EMBED_MODEL = st.sidebar.text_input("Модель эмбеддингов Ollama", OLLAMA_EMBED_MODEL_ENV)
OLLAMA_CHAT_MODEL = st.sidebar.text_input("Чат-модель Ollama", OLLAMA_CHAT_MODEL_ENV)
MONGO_URI = st.sidebar.text_input("MongoDB URI", MONGO_URI_ENV)
CHROMA_PERSIST_DIR = st.sidebar.text_input("Директория ChromaDB", CHROMA_PERSIST_DIR_ENV)

MONGO_DB_NAME = "constitutional_assistant_db"
MONGO_COLLECTION_NAME = "chat_history_ollama_v3"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ollama_setup_valid" not in st.session_state:
    st.session_state.ollama_setup_valid = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "current_rag_response_data" not in st.session_state:
    st.session_state.current_rag_response_data = None
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
if "db_collection" not in st.session_state:
    st.session_state.db_collection = None


def get_mongo_collection():
    if st.session_state.db_collection is None:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[MONGO_DB_NAME]
            st.session_state.db_collection = db[MONGO_COLLECTION_NAME]
        except Exception as e:
            st.sidebar.error(f"MongoDB Error: {e}")
            st.session_state.db_collection = None
            return None
    return st.session_state.db_collection


def save_message_to_db(message_doc):
    collection = get_mongo_collection()
    if collection is not None:
        try:
            collection.insert_one(message_doc)
        except Exception as e:
            st.error(f"DB Save Error: {e}")


def load_history_from_db(limit=50):
    collection = get_mongo_collection()
    history = []
    if collection is not None:
        try:
            db_messages = list(collection.find().sort("timestamp", DESCENDING).limit(limit))
            for doc in reversed(db_messages):
                history.append({
                    "role": doc.get("role"),
                    "content": doc.get("content"),
                    "id": str(doc.get("id", doc.get("_id"))),
                    "timestamp": doc.get("timestamp")
                })
        except Exception as e:
            st.error(f"DB History Load Error: {e}")
    return history


def check_ollama_availability():
    try:
        st.session_state.embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        st.session_state.llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        st.session_state.ollama_setup_valid = True
        st.sidebar.success(f"Ollama OK: {OLLAMA_EMBED_MODEL}, {OLLAMA_CHAT_MODEL}")
    except Exception as e:
        st.sidebar.error(f"Ollama Error: {e}")
        st.session_state.ollama_setup_valid = False
        st.session_state.embeddings = None
        st.session_state.llm = None


if st.sidebar.button("Применить настройки и Подключиться", key="apply_settings_button"):
    check_ollama_availability()
    get_mongo_collection()
    if not st.session_state.history_loaded and st.session_state.db_collection is not None:
        st.session_state.messages = load_history_from_db()
        st.session_state.history_loaded = True
    elif st.session_state.db_collection is None:
        st.session_state.messages = []
        st.session_state.history_loaded = False
    st.session_state.current_rag_response_data = None
    st.rerun()


def process_single_file_to_documents(uploaded_file_obj):
    docs = []
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file_obj.getvalue())
            tmp_file_path = tmp_file.name

        if uploaded_file_obj.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file_obj.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_file_path)
        else:
            return []

        loaded_documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(loaded_documents)
    except Exception as e:
        st.error(f"File Processing Error ({uploaded_file_obj.name}): {e}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    return docs


if not st.session_state.history_loaded and MONGO_URI:
    if st.session_state.db_collection is None:
        get_mongo_collection()
    if st.session_state.db_collection is not None:
        st.session_state.messages = load_history_from_db()
        st.session_state.history_loaded = True

uploaded_files_list = st.file_uploader(
    "Загрузите PDF/DOCX файлы с Конституцией",
    type=['pdf', 'docx'],
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files_list:
    if st.session_state.ollama_setup_valid and st.session_state.embeddings:
        all_processed_documents = []
        for single_file in uploaded_files_list:
            with st.spinner(f"Processing '{single_file.name}'..."):
                processed_docs = process_single_file_to_documents(single_file)
                if processed_docs:
                    all_processed_documents.extend(processed_docs)

        if all_processed_documents:
            with st.spinner("Creating/Updating ChromaDB..."):
                try:
                    if st.session_state.vectorstore is not None:
                        del st.session_state.vectorstore
                        st.session_state.vectorstore = None
                        gc.collect()
                        time.sleep(0.1)

                    if os.path.exists(CHROMA_PERSIST_DIR):
                        try:
                            shutil.rmtree(CHROMA_PERSIST_DIR)
                            st.info(f"Old ChromaDB directory '{CHROMA_PERSIST_DIR}' removed for update.")
                        except PermissionError:
                            st.error(
                                f"Permission error removing old ChromaDB. Close other apps using it or remove manually and retry.")
                        except Exception as e_del:
                            st.error(f"Error removing old ChromaDB: {e_del}")

                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=all_processed_documents,
                        embedding=st.session_state.embeddings,
                        persist_directory=CHROMA_PERSIST_DIR
                    )
                    st.success(f"Docs processed. ChromaDB at '{CHROMA_PERSIST_DIR}'.")
                    st.session_state.current_rag_response_data = None
                except Exception as e:
                    st.error(f"ChromaDB Error: {e}")
                    st.session_state.vectorstore = None
    else:
        st.warning("Setup Ollama & MongoDB in sidebar and click 'Apply & Connect'.")

if not uploaded_files_list and not st.session_state.vectorstore and \
        st.session_state.ollama_setup_valid and st.session_state.embeddings and \
        os.path.exists(CHROMA_PERSIST_DIR):
    with st.spinner(f"Loading existing ChromaDB from '{CHROMA_PERSIST_DIR}'..."):
        try:
            if st.session_state.vectorstore is not None:
                del st.session_state.vectorstore
                st.session_state.vectorstore = None
                gc.collect()
                time.sleep(0.1)
            st.session_state.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=st.session_state.embeddings
            )
            st.info(f"ChromaDB loaded.")
        except Exception as e:
            st.error(f"ChromaDB Load Error: {e}")
            st.session_state.vectorstore = None


def get_deeper_analysis_from_ollama(question, rag_answer, source_snippets_content, llm_instance):
    formatted_source_snippets = ""
    if source_snippets_content:
        for i, snippet in enumerate(source_snippets_content):
            formatted_source_snippets += f"Fragment {i + 1}:\n\"\"\"\n{snippet}\n\"\"\"\n\n"
    else:
        formatted_source_snippets = "Key fragments were not provided or extracted.\n"

    prompt_template = f"""Context of the Request:
User asked the following question about the Constitution of RK: "{question}"
Current answer based on extracted articles: "{rag_answer}"
Extracted key fragments from the Constitution:
{formatted_source_snippets}
Task for In-Depth Analysis:
Please provide a deeper analysis based EXCLUSIVELY on the question, answer, and constitutional fragments provided above. Your analysis should:
1. Briefly summarize the main legal aspects addressed in the answer and fragments as they apply to the question.
2. Explain how different parts of the extracted fragments are interconnected and related to the user's question.
3. If appropriate and the information is contained within the provided fragments, indicate possible general implications or interpretations (avoid external knowledge or assumptions not based on the text).
4. DO NOT repeat the original answer verbatim. Your task is to add depth and coherence to the understanding.
5. Avoid general phrases. Be specific, referring to the provided text.
6. The answer should be structured and easy to read.
In-Depth Analysis:"""
    try:
        if llm_instance:
            response = llm_instance.invoke(prompt_template)
            return response
        return "LLM for analysis is not available."
    except Exception as e:
        return f"Error generating in-depth analysis: {e}"


if st.session_state.vectorstore and st.session_state.ollama_setup_valid and st.session_state.llm:
    for msg_idx, message_doc in enumerate(st.session_state.messages):
        with st.chat_message(message_doc["role"]):
            content_data = message_doc.get("content", {})
            msg_id = message_doc.get("id", f"msg_{msg_idx}_{message_doc['role']}")

            if isinstance(content_data, str):
                st.write(content_data)
            else:
                st.write(content_data.get("answer", "No answer text."))
                if "sources" in content_data and content_data["sources"]:
                    with st.expander("Sources"):
                        for i, source_content in enumerate(content_data["sources"]):
                            st.markdown(f"**Source {i + 1}:**")
                            st.text_area(label=f"src_{msg_id}_{i}", value=source_content, height=100,
                                         disabled=True, label_visibility="collapsed",
                                         key=f"src_{msg_id}_{i}_{str(source_content)[:10]}")
                if "analysis" in content_data and content_data["analysis"]:
                    with st.expander("In-depth Analysis", expanded=True):
                        st.markdown(content_data["analysis"])

    if prompt := st.chat_input("Ask a question about the Constitution of RK"):
        user_message_id = str(datetime.now(timezone.utc).timestamp()) + f"_user_{len(st.session_state.messages)}"
        user_message_content = {"answer": prompt}
        user_msg_doc = {
            "role": "user", "content": user_message_content, "id": user_message_id,
            "timestamp": datetime.now(timezone.utc)
        }
        st.session_state.messages.append(user_msg_doc)
        save_message_to_db(user_msg_doc)

        with st.chat_message("assistant"):
            with st.spinner("Searching for an answer..."):
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm, chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    rag_response = qa_chain.invoke({"query": prompt})
                    result_text = rag_response.get("result", "Could not get an answer.")
                    source_documents = rag_response.get("source_documents", [])
                    sources_content = [doc.page_content for doc in source_documents]

                    st.session_state.current_rag_response_data = {
                        "question": prompt, "answer": result_text, "sources": sources_content
                    }

                    assistant_message_id = str(
                        datetime.now(timezone.utc).timestamp()) + f"_assistant_{len(st.session_state.messages)}"
                    assistant_message_content = {
                        "answer": result_text, "sources": sources_content, "analysis": None
                    }
                    assistant_msg_doc = {
                        "role": "assistant", "content": assistant_message_content, "id": assistant_message_id,
                        "timestamp": datetime.now(timezone.utc)
                    }
                    st.session_state.messages.append(assistant_msg_doc)
                    save_message_to_db(assistant_msg_doc)
                    st.rerun()

                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    err_msg_id = str(
                        datetime.now(timezone.utc).timestamp()) + f"_error_{len(st.session_state.messages)}"
                    error_assistant_msg_doc = {
                        "role": "assistant",
                        "content": {"answer": f"An error occurred: {e}", "sources": [], "analysis": None},
                        "id": err_msg_id, "timestamp": datetime.now(timezone.utc)
                    }
                    st.session_state.messages.append(error_assistant_msg_doc)
                    save_message_to_db(error_assistant_msg_doc)
                    st.rerun()

    if st.session_state.current_rag_response_data and \
            st.session_state.messages and \
            st.session_state.messages[-1]["role"] == "assistant" and \
            st.session_state.messages[-1]["content"].get("answer") and \
            not st.session_state.messages[-1]["content"].get("analysis"):

        if st.button("Get In-depth Analysis", key="deeper_analysis_button"):
            with st.spinner("Generating in-depth analysis..."):
                rag_data = st.session_state.current_rag_response_data
                analysis_text = get_deeper_analysis_from_ollama(
                    rag_data["question"], rag_data["answer"], rag_data["sources"], st.session_state.llm
                )

                last_assistant_message_index = -1
                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    if st.session_state.messages[i]["role"] == "assistant" and \
                            st.session_state.messages[i]["content"].get("answer") == rag_data[
                        "answer"]:  # Match the specific response
                        last_assistant_message_index = i
                        break

                if last_assistant_message_index != -1:
                    current_assistant_message_id = st.session_state.messages[last_assistant_message_index]["id"]
                    st.session_state.messages[last_assistant_message_index]["content"]["analysis"] = analysis_text

                    db_collection_instance = get_mongo_collection()
                    if db_collection_instance is not None:
                        try:
                            db_collection_instance.update_one(
                                {"id": current_assistant_message_id},
                                {"$set": {"content.analysis": analysis_text, "timestamp": datetime.now(timezone.utc)}}
                            )
                        except Exception as e:
                            st.error(f"DB Analysis Update Error: {e}")

                st.session_state.current_rag_response_data = None
                st.rerun()

elif not st.session_state.ollama_setup_valid:
    st.info(
        "Configure Ollama & MongoDB in the sidebar and click 'Apply Settings & Connect'. Ensure Ollama server is running.")
elif not st.session_state.vectorstore:
    st.info("Upload Constitution file(s) to begin, or check settings if the vector database did not load.")