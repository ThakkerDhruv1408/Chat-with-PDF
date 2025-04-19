# 📄 Chat with PDF — Conversational PDF Assistant

Transform static PDF documents into interactive conversational agents using **Retrieval-Augmented Generation (RAG)** powered by **Ollama models** like `mistral` and `nomic-embed-text`.

This project enables you to **ask questions about your PDFs**, and get intelligent responses based on the embedded content — all locally, securely, and efficiently.

---

## 🧠 How It Works

This system combines:

- **PDF Parsing**: Extract text content from PDF files
- **Text Embedding**: Embed text using `nomic-embed-text` for semantic understanding
- **Retrieval**: Search and retrieve relevant chunks based on user queries
- **Generation**: Use `mistral` to generate responses conditioned on retrieved context

The result is a lightweight RAG pipeline for chat-style interactions with document content.

---

## ✨ Features

- 💬 Chat with your PDFs like you're talking to an expert
- 🧩 RAG (Retrieval-Augmented Generation) architecture
- 🔍 Semantic search using local embedding models
- 🤖 Ollama integration with models like `mistral` and `nomic-embed-text`
- 🔒 Fully local and secure (no external API calls)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Chat-with-PDF.git
cd Chat-with-PDF
```
### 2. Install Dependencies
Make sure you have Python 3.9+ and Ollama installed.
```bash
pip install -r requirements.txt
```
### 3. Start Ollama and Pull Models
Make sure Ollama is running and then pull required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```
### 4. Run the App
```bash
python bot.py

```
---
Happy Codding!
