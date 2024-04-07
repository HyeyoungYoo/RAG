import streamlit as st

st.set_page_config(
    page_title="QuizGPT",
    page_icon="?",

)
st.title("privateGPT")
# 어떤 아티클을 사용할지 사용자가 결정하도록 한다.
with st.sidebar
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipidia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Uplod a .docx, .txt or .pdf",
            type=["pdf","txt","docx"],
        )
    else:
        topic = st.text_input("Name of the article")