import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Document Info Extraction", layout="wide")

st.title("📄 Document Information Extraction & QA", text_alignment='center')
st.divider()

col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    st.subheader("📝 Input Document")

    uploaded_file = st.file_uploader(
        "Upload receipt image",
        type=["png", "jpg", "jpeg"]
    )

    # Show initial small thumbnail (80x90 px) preview
    if uploaded_file:
        # Open the uploaded file as an image
        image = Image.open(uploaded_file)
        uploaded_file.seek(0)

        # Resize the image to 80x90 for preview
        small_width, small_height = 80, 90
        thumbnail_image = image.resize((small_width, small_height))

        # Display the small image thumbnail
        st.image(thumbnail_image, caption="Uploaded Receipt (Preview)", use_container_width=False)

    # EXTRACT BUTTON
    if st.button("Extract Information"):
        if uploaded_file is None:
            st.warning("Please upload an image")
        else:
            # Reset pointer before sending
            uploaded_file.seek(0)

            response = requests.post(
                f"{API_URL}/extract",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
            )

            if response.status_code == 200:
                result = response.json()

                st.session_state["extract_result"] = result
                st.session_state["extracted_text"] = result["text"]

            else:
                st.error("API error during extraction")


    # SHOW OCR TEXT (VERY USEFUL)
    if "extracted_text" in st.session_state:
        st.subheader("🧾 OCR Extracted Text")
        st.text(st.session_state["extracted_text"])

    # QUESTION SECTION
    st.subheader("💬 Ask a Question")

    question = st.text_input("Example: What is the total amount?")

    if st.button("Get Answer"):
        extracted_text = st.session_state.get("extracted_text", "")

        if extracted_text.strip() == "" or question.strip() == "":
            st.warning("Please extract data first and enter a question")
        else:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "text": extracted_text,
                    "question": question
                }
            )

            if response.status_code == 200:
                answer = response.json()["answer"]
                st.session_state["answer"] = answer
            else:
                st.error("API error during query")

    # SHOW ANSWER
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