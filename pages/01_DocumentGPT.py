from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="😎",
)

def embed_file(file):
    st.write(file)
    #업로드된 파일을 로컬의 캐시 폴더에 저장하여 loader로 부를 수 있도록 한다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path,"wb") as f:  #writable, binary로 파일 열기
        f.write(file_content) # 해당 파일에 내용 쓰기

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter().from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,  
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs,cached_embeddings) 
    retriever = vectorstore.as_retriever()
    return retriever

st.title("DocumentGPT")
#사용자에게 파일 업로드 요청

st.markdown("""
Welcome!
            
Use this chatbot to ask question to an AI about your files!
"""
)

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["txt","pdf","docx",]
)

if file:
   retriever = embed_file(file)
   s= retriever.invoke("winston")
   st.write(s)








