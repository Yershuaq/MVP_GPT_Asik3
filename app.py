import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os


st.set_page_config(page_title="Конституционный Ассистент", page_icon="📚")
st.title("🤖 Ассистент по Конституции РК")


if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

if not st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = st.text_input("Введите ваш OpenAI API ключ:", type="password")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2txtLoader(tmp_file_path)
    else:
        st.error("Поддерживаются только PDF и DOCX файлы")
        return None

    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    os.unlink(tmp_file_path)
    return splits


uploaded_file = st.file_uploader("Загрузите PDF или DOCX файл с Конституцией", type=['pdf', 'docx'])

if uploaded_file and st.session_state.OPENAI_API_KEY:
    with st.spinner("Обработка документа..."):
        splits = process_document(uploaded_file)
        if splits:
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.OPENAI_API_KEY)
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success("Документ успешно обработан!")

if st.session_state.vectorstore:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("Источники"):
                    for source in message["sources"]:
                        st.write(source)

    if prompt := st.chat_input("Задайте вопрос о Конституции РК"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0,
                        openai_api_key=st.session_state.OPENAI_API_KEY
                    ),
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                )
                
                response = qa_chain({"query": prompt})
                
                st.write(response["result"])
                
                with st.expander("Источники"):
                    for doc in response.get("source_documents", []):
                        st.write(doc.page_content)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["result"],
                    "sources": [doc.page_content for doc in response.get("source_documents", [])]
                }) 