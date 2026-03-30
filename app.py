# app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Info Extraction", layout="centered")

st.title("📄 Document Information Extraction & QA")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Document Text")

text_input = st.text_area("Paste receipt text here:", height=150)


# -----------------------------
# Extract Button
# -----------------------------
if st.button("Extract Information"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        response = requests.post(
            f"{API_URL}/extract",
            json={"text": text_input}
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("📊 Extracted Data")
            st.json(result["data"])

            st.subheader("🔍 Predictions")
            st.write(result["predictions"])
        else:
            st.error("API error")


# -----------------------------
# Q&A Section
# -----------------------------
st.subheader("Ask a Question")

question = st.text_input("Example: What is the total amount?")

if st.button("Get Answer"):
    if text_input.strip() == "" or question.strip() == "":
        st.warning("Please enter both text and question")
    else:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "text": text_input,
                "question": question
            }
        )

        if response.status_code == 200:
            answer = response.json()["answer"]

            st.subheader("💬 Answer")
            st.write(answer)
        else:
            st.error("API error")