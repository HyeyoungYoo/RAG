import json
from operator import rshift
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser

class JsonOutputParser(BaseOutputParser): 

    def parse(self, text): 
        text = text.replace("```","").replace("json","")
        return json.loads(text)  #json파일을 pyton 오브젝트로 바꾸어줌

    
output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="?",
)
st.title("QuizGPT")

file = None
topic = None

llm = ChatOpenAI(
    temperature=0.1,
    model = "gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
You are a helpful assistant that is role playing as a teacher.
        
Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

Each question should have 4 answers, three of them must be incorrect and one should be correct.
        
Use (o) to signal the correct answer.

if context language is Korean, you should make it by Korean
        
Question examples:
        
Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)
        
Question: What is the capital or Georgia?
Answers: Baku|Tbilisi(o)|Manila|Beirut
        
Question: When was Avatar released?
Answers: 2007|2001|2009(o)|1998
        
Question: Who was Julius Caesar?
Answers: A Roman Emperor(o)|Painter|Actor|Model
        
    Example Output:
     
    ```json{{ "questions": [ 
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}```
Your turn!
        
Context: {context}
"""),
    ]
)
questions_chain = {"context": format_docs} | questions_prompt | llm #질문 생성

formatting_chain = formatting_prompt | llm #질문으로 json 생성

@st.cache_data(show_spinner="Loadinging file...") 
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path,"wb") as f: 
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder( 
        separator="\n",
        chunk_size=600,  
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path) 
    docs = loader.load_and_split(text_splitter=splitter) 
    return docs

@st.cache_data(show_spinner="Making Quiz...") #자꾸 체인 전체를 호출하지 않기 위해
def run_quiz_chain(_docs, topic):                   #캐시에 GPT로부터 받은 결과물을 저장
    #단 _docs이므로 첫번째가 유일하게 실행되는 것이고 항상 첫번째 실행값만 반환될 것임.
    #document가 변경되더라도..so 이를 막기위해 hashing을 위한 다른 파라미터를 추가해야 함.
    #topic이 바뀌면 이 함수는 재실행된다.
    chain = {"context":questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...") #위키 사전도 캐싱
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5,lang="en") #가장 상위 5개 문서만 원한다는 뜻, 한국어는 lang="ko"
    docs = retriever.get_relevant_documents(term)
    return docs
#st.cache_data는 아래 데코레이트 한 함수를 살펴보는 것.
# wiki_search(term) 에서 파라미터인 term이 hash된다.
# hashsms 들어오는 데이터의 서명을 생성함을 의미함.
#그래서 다시 이 함수를 호출하게 되면 데이터의 서명이 변경되지 않은 경우
#streamlit이 이 함수가 실행되면 안된다는 것을 알아차리고 재실행 없이 이전값을 리턴해줌.
#주의점은 해싱이 불가능한 데이터타입이 있는 점.(사용자 정의 객체나 일부 변경 가능한 데이터 구조(mutable data structures)는 기본적으로 해시가 불가능)이 경우 streamlit이 해시에러발생
#함수의 매개변수 이름이 _로 시작하는 경우, Streamlit은 그 매개변수를 해싱 과정에서 제외시키다. 해싱할 수 없거나 해싱하기에 부적합한 큰 데이터셋이나 복잡한 객체를 함수의 입력으로 사용해야 할 때 유용. 단, 이 매개변수값은 변경되어도 캐싱되지 않아 문제발생 가능


# 어떤 아티클을 사용할지 사용자가 결정하도록 한다.
with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Uplod a .docx, .txt or .pdf",
            type=["pdf","txt","docx"],
        )
        if file:
            docs=split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        st.spinner(text="Searching...")
        if topic:
            docs = wiki_search(topic)
           

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles od files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.

    """
    )
else:

    start = st.button("Generate Quiz")

    if start:
    #    questions_response = questions_chain.invoke(docs)
    #    formatting_response = formatting_chain.invoke({"context" : questions_response.content}) --> 이 내용을 아래 한줄로 간단히 표현 
        reponse = run_quiz_chain(docs, topic if topic else file.name) #topic 이 존재하는 경우는 topic을. 파일을 선택한 경우는 파일명을 넘겨주자    
        st.write(reponse)
