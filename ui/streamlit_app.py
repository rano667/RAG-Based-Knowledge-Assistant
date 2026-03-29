import streamlit as st
import requests

st.title("RAG Invoice Assistant")

query = st.text_input("Ask something:")

if st.button("Submit"):
    res = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"query": query}
    )
    
    st.write(res.json()["answer"])