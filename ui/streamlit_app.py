import streamlit as st
import requests

st.title("RAG Invoice Assistant")

query = st.text_input("Ask something:")

if st.button("Submit"):
    with st.spinner("⏳ Processing your request... this may take up to a minute"):
        res = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"query": query}
        )
    
    st.success("✅ Done")
    st.write(res.json()["answer"])