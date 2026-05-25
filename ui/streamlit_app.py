import streamlit as st
import requests

from config import API_URL

sample_questions = [
    "Custom Question...",
    "What items are in invoice 0012820?",
    "What are the products in invoice 1213?",
    "Who is the buyer in invoice 0012820?",
    "What is the total due in invoice 1213?",
    "Which products were ordered by Caitlin Roberts?"
]

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Invoice Assistant",
    page_icon="📄",
    layout="centered"
)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📌 Project Info")

    st.markdown(
        """
        ### 🔗 Links
        
        - [GitHub Repository](https://github.com/rano667/RAG-Based-Knowledge-Assistant)
        - [Dataset Directory](https://github.com/rano667/RAG-Based-Knowledge-Assistant/tree/main/data)

        ### 🧠 Tech Stack
        
        - FastAPI
        - FAISS
        - Groq LLM
        - HuggingFace Embeddings
        - Docker
        - AWS ECS + ECR
        - RAGAS Evaluation

        ### ☁️ Deployment
        
        Hosted on AWS ECS Fargate
        """
    )

# -----------------------------
# MAIN UI
# -----------------------------
st.title("📄 RAG Invoice Assistant")

st.markdown(
    """
    Ask questions about invoice PDFs using a production-style
    Retrieval-Augmented Generation (RAG) pipeline.
    """
)

selected_question = st.selectbox(
    "Ask something about the invoices:",
    sample_questions
)

# Only show text input if custom selected
if selected_question == "Custom Question...":
    query = st.text_input(
        "Enter your custom question:"
    )
else:
    query = selected_question

# -----------------------------
# SUBMIT BUTTON
# -----------------------------
if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a query.")
    
    else:
        with st.spinner(
            "⏳ Processing your request... this may take up to a minute"
        ):

            try:
                res = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=120
                )

                data = res.json()

                st.success("✅ Done")

                st.markdown("### 📌 Answer")

                formatted_answer = (
                    data["answer"]
                    .replace("• ", "\n• ")
                    .strip()
                )

                st.markdown(formatted_answer)

                # Optional metadata section
                with st.expander("🔍 Retrieval Metadata"):

                    st.write(
                        "Retrieved Chunk IDs:",
                        data.get("retrieved_chunk_ids")
                    )

                    st.write(
                        "Sources:",
                        data.get("retrieved_sources")
                    )

                    st.write(
                        "Latency:",
                        f"{data.get('latency_seconds', 0):.2f} sec"
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
