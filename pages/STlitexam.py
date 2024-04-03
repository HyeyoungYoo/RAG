import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🥪"
)

st.title("Document GPT")
# streamlit의 Session State : 코드가 반복 실행되어도 지워지지 않는 저장공간
if "messages" not in st.session_state: #st.session_staterk가
                                       #messages라는 key를 가지고 있지않다면
    st.session_state["messages"]=[]    #그때 빈 list로 initialize
                                       #key 가 있으면 아무것도 하지마(messages를 유지하게)
#st.write(st.session_state["messages"])

def send_message(message,role,save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message":message,"role":role})

for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )

message = st.chat_input("Send a message to the ai")

if message:
    send_message(message,"human")
    time.sleep(2)
    send_message(f"You said: {message}","ai")

    with st.sidebar:
        st.write(st.session_state)

# with st.chat_message("human"):
#     st.write("Helloooooooooooooo")

# with st.chat_message("ai"):
#     st.write("how are you")

# with st.status("Embedding file...",expanded=True) as status:
#     time.sleep(2)
#     st.write("Getting the file")
#     time.sleep(2)
#     st.write("Embedding the file")
#     time.sleep(2)
#     st.write("Caching the file")
#     status.update(label="Error",state="error")








