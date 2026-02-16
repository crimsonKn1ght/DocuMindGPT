# DocuMindGPT

<p align="center">
  <a href="https://github.com/crimsonKn1ght/DocuMindGPT/stargazers">
    <img src="https://img.shields.io/github/stars/crimsonKn1ght/DocuMindGPT?style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/crimsonKn1ght/DocuMindGPT/network/members">
    <img src="https://img.shields.io/github/forks/crimsonKn1ght/DocuMindGPT?style=for-the-badge" alt="GitHub forks">
  </a>
  <a href="https://github.com/crimsonKn1ght/DocuMindGPT/graphs/commit-activity">
    <img src="https://img.shields.io/maintenance/yes/2026?style=for-the-badge" alt="Maintained">
  </a>
  <a href="https://github.com/crimsonKn1ght/DocuMindGPT">
    <img src="https://img.shields.io/github/languages/top/crimsonKn1ght/DocuMindGPT?style=for-the-badge" alt="Language">
  </a>
</p>

DocuMindGPT is a document-grounded Q&A CLI that uses RAG (Retrieval-Augmented Generation) to answer questions based on your PDFs or text files. It features a built-in evaluation agent that scores answers for hallucinations and relevance.

## Flowchart

<img width="3882" height="8192" alt="flowchart" src="https://github.com/user-attachments/assets/9b75be38-fe67-4fc4-9bde-3bf36ae758c6" />


## Features

* **Ingestion**: Chunk, embed, and store documents in Supabase (pgvector).
* **RAG Chat**: Query your knowledge base using Gemini 2.5 Flash.
* **Auto-Evaluation**: Every answer is audited by a secondary AI agent for accuracy (Pass/Fail verdict).

## Prerequisites

* Python 3.10+
* A [Supabase](https://supabase.com/) project
* A [Google Gemini API Key](https://ai.google.dev/)

## Setup

1.  **Database**: Run the contents of `setup.sql` in your Supabase SQL Editor to enable `pgvector` and create the required tables.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**: Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_google_api_key
    SUPABASE_URL=your_supabase_project_url
    SUPABASE_KEY=your_supabase_anon_key
    ```

## Usage

### 1. Upload a Document
Ingest a file (PDF or text) into the vector store.
```bash
python main.py upload path/to/document.pdf
