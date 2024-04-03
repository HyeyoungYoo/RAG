import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime

st.title("Hello World")
st.subheader("Welcome to streamlit")
st.markdown("""### I Love it """)
st.write("{x:1}")

today = datetime.today().strftime("%d/%m/%y, %H:%M:%S")
st.title(today)

model = st.selectbox("Choose your model",
                         (
                            "GPT-3",
                            "GPT-4",
                            "GPT-5"
                        )
                    )
st.write(model)

if model=="GPT-3":
    st.write("Cheap")
elif model=="GPT-4":
    st.write("not Cheap")
    name = st.text_input("What's your name?")
    st.write(name)
    value = st.slider("temperature",min_value=0.1, max_value=1.0 )
    st.write(value)

    with st.sidebar:
    st.title("sidebar title")

    st.text_input("ㅍㅍㅍㅍ")

tab_one, tab_two, tab_three = st.tabs(["A","B","C"])

'''
with tab_one:
    st.write("a")
with tab_two:
    st.write("b")
with tab_three:
    st.write("c")'''
