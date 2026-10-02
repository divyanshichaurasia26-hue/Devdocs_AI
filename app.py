import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from io import BytesIO

from rag import create_chunks, retrieve_relevant_chunks


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DevDocs AI",
    page_icon="💻",
    layout="wide"
)


# =====================================================
# APPLICATION TITLE
# =====================================================

st.title("💻 DevDocs AI")

st.subheader(
    "GenAI-Powered Software Documentation Assistant"
)

st.write(
    "Upload technical documentation and ask questions "
    "about it using Generative AI."
)


# =====================================================
# GEMINI API KEY
# =====================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "Gemini API key was not found. "
        "Please check your .env file."
    )

    st.stop()


# =====================================================
# GEMINI CLIENT
# =====================================================

client = genai.Client(
    api_key=api_key
)


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📄 Upload a technical document",
    type=[
        "txt",
        "pdf"
    ]
)


# =====================================================
# DOCUMENT PROCESSING
# =====================================================

if uploaded_file is not None:

    st.success(
        f"✅ Uploaded: {uploaded_file.name}"
    )

    document_text = ""


    # =================================================
    # PDF PROCESSING
    # =================================================

    if uploaded_file.name.lower().endswith(
        ".pdf"
    ):

        try:

            pdf_bytes = uploaded_file.getvalue()

            pdf_stream = BytesIO(
                pdf_bytes
            )

            pdf_reader = PdfReader(
                pdf_stream
            )

            for page in pdf_reader.pages:

                text = page.extract_text()

                if text:

                    document_text += (
                        text + "\n"
                    )

        except Exception as error:

            st.error(
                "❌ This PDF could not be read."
            )

            with st.expander(
                "Technical error details"
            ):

                st.code(
                    str(error)
                )

            st.stop()


    # =================================================
    # TXT PROCESSING
    # =================================================

    else:

        try:

            document_text = (
                uploaded_file
                .getvalue()
                .decode("utf-8")
            )

        except UnicodeDecodeError:

            st.error(
                "❌ The TXT file could not be decoded."
            )

            st.stop()


    # =================================================
    # CHECK DOCUMENT
    # =================================================

    if not document_text.strip():

        st.warning(
            "⚠️ No readable text was found."
        )

        st.stop()


    # =================================================
    # DOCUMENT CONTENT
    # =================================================

    st.subheader(
        "📄 Document Content"
    )

    st.text_area(
        "Extracted document text",
        document_text,
        height=300
    )


    # =================================================
    # CREATE CHUNKS
    # =================================================

    chunks = create_chunks(
        document_text
    )

    st.subheader(
        "🧩 Document Processing"
    )

    st.success(
        f"Document divided into "
        f"{len(chunks)} chunks."
    )


    # =================================================
    # VIEW CHUNKS
    # =================================================

    with st.expander(
        "View document chunks"
    ):

        for i, chunk in enumerate(
            chunks
        ):

            st.markdown(
                f"### Chunk {i + 1}"
            )

            st.write(
                chunk
            )

            st.divider()


    # =================================================
    # QUESTION SECTION
    # =================================================

    st.subheader(
        "💬 Ask DevDocs AI"
    )

    question = st.text_input(
        "Ask a question about your document:",
        placeholder=(
            "Example: What is the purpose "
            "of Kubernetes?"
        )
    )


    # =================================================
    # RAG RETRIEVAL
    # =================================================

    if question:

        relevant_chunks = (
            retrieve_relevant_chunks(
                question,
                chunks,
                top_k=3
            )
        )


        # =================================================
        # DISPLAY RETRIEVED INFORMATION
        # =================================================

        st.subheader(
            "🔎 Retrieved Information"
        )

        for result in relevant_chunks:

            chunk_index = (
                result["chunk_index"]
            )

            similarity = (
                result["similarity"]
            )

            chunk_text = (
                result["text"]
            )

            st.markdown(
                f"**Chunk {chunk_index + 1}**  "
                f"**Similarity: {similarity:.3f}**"
            )

            st.write(
                chunk_text
            )

            st.divider()


        # =================================================
        # CREATE CONTEXT
        # =================================================

        context = "\n\n".join(
            result["text"]
            for result in relevant_chunks
        )


        # =================================================
        # GEMINI PROMPT
        # =================================================

        prompt = f"""
You are DevDocs AI, an AI assistant for
software developers.

Answer the user's question using the retrieved
information from the uploaded document.

RETRIEVED DOCUMENT INFORMATION:

{context}

USER QUESTION:

{question}

Instructions:

1. Use the retrieved document information
   as the primary source.

2. Give a clear and concise answer.

3. Do not invent information.

4. If the answer cannot be found in the
   retrieved information, say:

"I couldn't find that information in the
uploaded document."

5. Do not use unrelated outside information.
"""


        # =================================================
        # AI ANSWER
        # =================================================

        st.subheader(
            "🤖 AI Answer"
        )

        try:

            response = (
                client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )
            )

            if response.text:

                st.write(
                    response.text
                )

            else:

                st.warning(
                    "The AI returned an empty response."
                )

        except Exception as error:

            st.error(
                "⚠️ The Gemini AI service is "
                "temporarily unavailable."
            )

            with st.expander(
                "Technical error details"
            ):

                st.code(
                    str(error)
                )