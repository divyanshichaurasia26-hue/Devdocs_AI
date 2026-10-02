# 💻 DevDocs AI

## GenAI-Powered Software Documentation Assistant

DevDocs AI is a Retrieval-Augmented Generation (RAG) based application that allows users to upload technical documents and ask questions about their content.

The application retrieves the most relevant information from the uploaded document using semantic similarity search and uses Google Gemini to generate context-aware answers.

---

## 🚀 Live Demo

**Try DevDocs AI online:**

👉 **Streamlit App:** YOUR_STREAMLIT_APP_URL

The application is deployed using Streamlit Community Cloud and can be accessed directly from a web browser.

---

## ✨ Features

- 📄 Upload PDF and TXT documents
- 📖 Extract text from uploaded documents
- 🧩 Split documents into smaller chunks
- 🧠 Generate semantic embeddings using Sentence Transformers
- 🔎 Retrieve relevant information using cosine similarity
- 📊 Display retrieved information and similarity scores
- 🤖 Generate answers using Google Gemini
- 🌐 Cloud deployment using Streamlit Community Cloud
- 🔐 Secure API key management using Streamlit Secrets

---

## 🧠 How It Works

DevDocs AI follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
Upload Document
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Generate Embeddings
      ↓
Semantic Similarity Search
      ↓
Retrieve Relevant Information
      ↓
Send Retrieved Context to Gemini
      ↓
Generate AI Answer
