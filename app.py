import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="Конституционный Ассистент РК на Ollama", page_icon="📚")
st.title("🤖 Ассистент по Конституции РК (на базе Ollama)")

OLLAMA_BASE_URL = st.sidebar.text_input("Адрес сервера Ollama (если не localhost:11434)", "http://localhost:11434")
OLLAMA_EMBED_MODEL = st.sidebar.text_input("Модель эмбеддингов Ollama", "nomic-embed-text")
OLLAMA_CHAT_MODEL = st.sidebar.text_input("Чат-модель Ollama", "llama2")

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


def check_ollama_availability():
    try:
        st.session_state.embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        st.session_state.llm = ChatOllama(
            model=OLLAMA_CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )
        st.session_state.ollama_setup_valid = True
        st.sidebar.success(f"Ollama настроен: Embed: {OLLAMA_EMBED_MODEL}, Chat: {OLLAMA_CHAT_MODEL}")
    except Exception as e:
        st.sidebar.error(f"Ошибка подключения к Ollama или модели не найдены: {e}")
        st.session_state.ollama_setup_valid = False
        st.session_state.embeddings = None
        st.session_state.llm = None


if st.sidebar.button("Применить настройки Ollama"):
    check_ollama_availability()


def process_document(file):
    tmp_file_path = ""
    try:
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
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        return splits
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


uploaded_file = st.file_uploader("Загрузите PDF или DOCX файл с Конституцией", type=['pdf', 'docx'])

if uploaded_file and st.session_state.ollama_setup_valid and st.session_state.embeddings:
    with st.spinner(f"Обработка документа '{uploaded_file.name}'..."):
        splits = process_document(uploaded_file)
        if splits:
            try:
                st.session_state.vectorstore = FAISS.from_documents(splits, st.session_state.embeddings)
                st.success(f"Документ '{uploaded_file.name}' успешно обработан и добавлен в базу знаний!")
            except Exception as e:
                st.error(f"Ошибка при создании векторного хранилища: {e}")
                st.session_state.vectorstore = None
        else:
            st.error("Не удалось получить части текста из документа.")

elif uploaded_file and (not st.session_state.ollama_setup_valid or not st.session_state.embeddings):
    st.warning("Пожалуйста, настройте и примените параметры Ollama в боковой панели перед загрузкой файла.")

if st.session_state.vectorstore and st.session_state.ollama_setup_valid and st.session_state.llm:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Источники"):
                    for i, source_content in enumerate(message["sources"]):
                        st.markdown(f"**Источник {i + 1}:**")
                        st.text_area(label=f"src_{i}_{message['role']}_{i}", value=source_content, height=100,
                                     disabled=True, label_visibility="collapsed",
                                     key=f"src_{message['role']}_{i}_{source_content[:10]}")

    if prompt := st.chat_input("Задайте вопрос о Конституции РK"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Идет поиск ответа..."):
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )

                    response = qa_chain.invoke({"query": prompt})  # Используйте .invoke для новых версий Langchain
                    result_text = response.get("result", "Не удалось получить ответ.")
                    source_documents = response.get("source_documents", [])

                    st.write(result_text)

                    sources_content = []
                    if source_documents:
                        with st.expander("Источники ответа"):
                            for i, doc in enumerate(source_documents):
                                st.markdown(f"**Источник {i + 1} (часть документа):**")
                                st.text_area(label=f"res_src_{i}", value=doc.page_content, height=100, disabled=True,
                                             label_visibility="collapsed", key=f"res_src_{i}_{doc.page_content[:10]}")
                                sources_content.append(doc.page_content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result_text,
                        "sources": sources_content
                    })
                except Exception as e:
                    st.error(f"Ошибка при получении ответа от LLM: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Произошла ошибка: {e}",
                        "sources": []
                    })
elif not st.session_state.ollama_setup_valid:
    st.info(
        "Пожалуйста, настройте параметры Ollama в боковой панели слева и нажмите 'Применить настройки Ollama'. Убедитесь, что ваш сервер Ollama запущен с необходимыми моделями.")
elif not st.session_state.vectorstore and uploaded_file:
    st.info("Идет обработка файла или произошла ошибка при настройке Ollama. Проверьте настройки и попробуйте снова.")
elif not st.session_state.vectorstore:
    st.info("Пожалуйста, загрузите документ с Конституцией для начала работы.")