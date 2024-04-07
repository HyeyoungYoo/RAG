#streamlit Challenge -4/5 +
# memory 달기 -- 미완성

from langchain.document_loaders import UnstructuredFileLoader 
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="😎",
)

api_key = ""

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args,**kwargs):
        self.message_box = st.empty() 
        super.load_memory()

    def on_llm_end(self, *args,**kwargs):
        save_message(self.message,"ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,  
    callbacks=[
        ChatCallbackHandler(),
    ],
    api_key=api_key
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path,"wb") as f:
        f.write(file_content)

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

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
       save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False,)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]

def run_chain(inputs):
    result = chain.invoke(inputs)
    print(result.content)
    memory.save_context(
        {"inputs":inputs},
        {"outputs":"".join(result.content)},
    )
    
memory = ConversationBufferMemory(
    llm=llm,
    return_messages=True,
    max_token_limit=200,
    memory_key="chat_history"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         Answer the question using ONLY the following context. If you don't know the answer just say you don't know, don't make anything up.

         Context: {context}
        """,),
        MessagesPlaceholder(variable_name="chat_history"),
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

with st.sidebar:
    api_key = st.text_input("Insert your API-KEY")
    if api_key:
       st.write("API-Key is entered")
    
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["txt","pdf","docx",]
    )


if file and api_key:
   retriever = embed_file(file)

   send_message("I'm ready! Ask away!","ai", save=False)
   paint_history()
   message = st.chat_input("Ask anything about your file...")
   if message:
        send_message(message,"human")
        chain = {                       
            "context":retriever | RunnableLambda(format_docs), 
            "question":RunnablePassthrough(),
            "chat_history": memory.load_memory_variables({}),
        } | prompt | llm
        with st.chat_message("ai"):
            run_chain(message)
            print("memory==",memory.load_memory_variables({}))
else: 
    st.session_state["messages"] = []








