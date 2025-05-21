
import streamlit as st
from temp import (
    pdf_to_text, text_to_text, excel_to_text,
    setup_query_engine, handle_chat
)


st.set_page_config(page_title="HELV", layout="wide")
st.title("HELV")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF, TXT, or Excel file", type=["pdf", "txt", "xlsx", "xls"])

    query_engine = None
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            content = pdf_to_text(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            content = text_to_text(uploaded_file)
        else:
            content = excel_to_text(uploaded_file)

        query_engine = setup_query_engine(content)
        st.success("Document uploaded and indexed!")
    else:
        content = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.chat_input("Type your message here...")

if prompt:
    st.session_state.chat_history.append(("user", prompt))
    response = handle_chat(prompt, query_engine)
    st.session_state.chat_history.append(("bot", response))
#st.write("Chat History Debug:", st.session_state.chat_history)

for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)
