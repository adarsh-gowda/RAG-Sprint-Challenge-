# RAG Sprint Challenge – AI Engineering Assignment

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for analyzing and comparing financial filings (10-K/Annual reports) of Google, Microsoft, and NVIDIA across 2022, 2023, and 2024.

The system extracts structured insights such as revenue, R&D spending, margins, and other financial metrics, while citing sources with company name, year, excerpt, and page number (if available).

## Features
- Document Parsing: Supports HTML, XHTML, and XML filings (SEC/EDGAR style).
- Chunking & Embedding: Uses LangChain with OpenAIEmbeddings for semantic search.
- RAG Pipeline: Combines retrieval + ChatOpenAI to answer queries grounded in filings.
- Source Attribution: Each answer includes company name, year, excerpt, and page reference.
- Comparison Queries: Handles multi-year, multi-company comparisons (e.g., revenue growth, R&D % of revenue, operating margins).

## Tech Stack
- Python 3.10+
- LangChain
- OpenAI API (gpt-4 model)
- BeautifulSoup4 (for document parsing)
- FAISS (vector store for retrieval)
- Pandas (for structured outputs & Excel export)


## Setup & Usage

### 1. Clone Repo
```bash
git clone https://github.com/adarsh-gowda/RAG-Sprint-Challenge-.git
cd RAG-Sprint-Challenge-
```

### 2. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

```

### 4. Add API Key
```bash
OPENAI_API_KEY=your_openai_api_key_here

```
### 5.Run script
```bash
python main.py

```

# Example test queries
    test_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "What percentage of Google's 2023 revenue came from advertising?",
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
    ]
