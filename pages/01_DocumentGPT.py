from email import message
from uuid import UUID
from langchain.document_loaders import UnstructuredFileLoader 
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="😎",
)

class ChatCallbackHandler(BaseCallbackHandler): #callback handeler는 각종 event들을 listen하는 function들을 갖는다.

    message = ""

    def on_llm_start(self, *args,**kwargs): #*args와 **kwarg로 수많은 argument(1,2,3,...) 및 keyword argument(a=1,b=2,c=3...)를 받을 수 있음
        self.message_box = st.empty()  #빈 위젯 제공

    def on_llm_end(self, *args,**kwargs):
        save_message(self.message,"ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,  #callback handler로 LLM의 답변 과정이 화면에 보일 수 일도록 함.
                    #ChatOpenAI는 가능.일부 오래된 모델은 이 기능을 지원X 
    callbacks=[
        ChatCallbackHandler(),
    ]
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding file...") #streamlit의 cache_data함수는 업로드된 파일이 동일하면 이 함수를 재실행하지 않고 기존의 값을 다시반환
def embed_file(file):
    #업로드된 파일을 로컬의 캐시 폴더에 저장하여 loader로 부를 수 있도록 한다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path,"wb") as f:  #writable, binary로 파일 열기
        f.write(file_content) # 해당 파일에 내용 쓰기

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}") #업로드 파일이름으로 캐시디렉토리 만들어
    splitter = CharacterTextSplitter().from_tiktoken_encoder( #spliter 생성
        separator="\n",
        chunk_size=600,  
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path) #업로드된 파일을 loader로 가져와
    docs = loader.load_and_split(text_splitter=splitter) #loader에 올려진 파일을 splitter로 쪼개 []
    embeddings = OpenAIEmbeddings() #임베딩스 생성
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) # 주어진 바이트 저장소(embeddings)에서 임베딩 데이터를 로드하여, 이를 지정된 캐시 디렉터리(cache_dir)에 캐싱하는 과정
    vectorstore = FAISS.from_documents(docs,cached_embeddings) #FAISS 라이브러리를 사용하여 문서(docs) 컬렉션으로부터 벡터 저장소(vectorstore)를 생성하는 과정을 설정. docs는 벡터화하려는 문서들의 컬렉션이며, cached_embeddings는 문서들을 벡터화할 때 사용할 임베딩 데이터. 함수의 결과는 벡터의 저장소 vectorstore라는 변수에 저장
    retriever = vectorstore.as_retriever() #사용자가 요청하는 쿼리에 대해 가장 관련성 높은 문서나 데이터 포인트를 vectorstore에서 검색해 반환
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
       save_message(message, role)

#히스토리를 그려보는 함수
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False,)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         Answer the question using ONLY the following context. If you don't know the answer just say you don't know, don't make anything up.

         Context: {context}
        """,
        ),
        ("human","{question}"),          
    ]  
)

st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask question to an AI about your files!
            
Upload your files on the sidebar!
"""
)
#사용자에게 파일 업로드 요청 at sidebar
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["txt","pdf","docx",]
    )

if file:
   retriever = embed_file(file)

   send_message("I'm ready! Ask away!","ai", save=False)
   paint_history()
   message = st.chat_input("Ask anything about your file...")
   if message:
        send_message(message,"human")
        chain = {                       
            "context":retriever | RunnableLambda(format_docs), 
            "question":RunnablePassthrough()
        } | prompt | llm
        #docs = retriever.invoke(message) #LangChain은 chain의 input을 이용해 retriever를 자동으로 invoke하므로 필요X
        with st.chat_message("ai"):
            chain.invoke(message)


else: 
    st.session_state["messages"] = [] #업로드 파일이 바뀌거나 파일을 삭제하면 저장했던 대화 세션 초기화








