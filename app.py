import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Info Extraction", layout="wide")

st.title("📄 Document Information Extraction & QA")

col1, col2 = st.columns(2)


# LEFT COLUMN
with col1:
    st.subheader("📝 Input Document")

    text_input = st.text_area("Paste receipt text here:", height=200)

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

                # Save result in session (important!)
                st.session_state["extract_result"] = result
            else:
                st.error("API error")

    st.divider()

    st.subheader("💬 Ask a Question")

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
                st.session_state["answer"] = answer
            else:
                st.error("API error")

    # Show answer
    if "answer" in st.session_state:
        st.subheader("📢 Answer")
        st.write(st.session_state["answer"])

# RIGHT COLUMN
with col2:
    st.subheader("📊 Extracted Results")

    if "extract_result" in st.session_state:
        result = st.session_state["extract_result"]

        st.markdown("### Structured Data")
        st.json(result["data"])

        st.markdown("### Token Predictions")
        st.write(result["predictions"])
    else:
        st.info("Run extraction to see results")