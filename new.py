import os
import json
from dotenv import load_dotenv
from io import StringIO
# Load environment variables
load_dotenv()

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from bs4 import BeautifulSoup
import pandas as pd
from langchain.docstore.document import Document
from unstructured.partition.html import partition_html
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import pdfplumber

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data1"
VECTOR_STORE_FILE = "faiss_index1"

# -----------------------------
# Initialize Gemini
# -----------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
import os
import pandas as pd
from bs4 import BeautifulSoup
from langchain.docstore.document import Document


def load_documents(data_dir):
    all_docs = []

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path) or not filename.lower().endswith(".pdf"):
            continue

        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # --- Extract text ---
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1, "type": "text"}
                        )
                        all_docs.append(doc)

                    # --- Extract tables ---
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        try:
                            # Convert each table to a string
                            table_text = "\n".join(
                                [" | ".join([str(cell) if cell else "" for cell in row]) for row in table]
                            )
                            if table_text.strip():
                                doc = Document(
                                    page_content=table_text,
                                    metadata={"source": filename, "page": i + 1, "type": "table", "table_index": t_idx}
                                )
                                all_docs.append(doc)
                        except Exception as e:
                            print(f"⚠️ Error parsing table on page {i+1} of {filename}: {e}")

            print(f"✅ Loaded {len(pdf.pages)} pages (with text + tables) from {filename}")

        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")

    print(f"\n📄 Total documents loaded: {len(all_docs)}")
    return all_docs
# -----------------------------
# VECTOR STORE
# -----------------------------
def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=os.getenv("GEMINI_API_KEY"))  # Gemini embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_FILE)
    return vectorstore

def preload_vector_store():
    """Load FAISS if available, else create from scratch."""
    if os.path.exists(VECTOR_STORE_FILE):
        print("✅ Vector store already exists. Loading...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=GEMINI_API_KEY
        )
        return FAISS.load_local(VECTOR_STORE_FILE, embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚠️ Vector store not found. Creating new one from data folder...")
        docs = load_documents(DATA_DIR)
        return create_vector_store(docs)

# -----------------------------
# PROMPT TEMPLATES
# -----------------------------
comparative_prompt = PromptTemplate(
    input_variables=["company_data", "query"],
    template="""
You are a financial analysis agent.
You are given extracted text snippets for multiple companies: {company_data}
Answer the following question by:
1. Comparing the relevant metrics across companies.
2. Returning a JSON with keys: query, answer, reasoning, sub_queries, sources each should be separate by new line.
3. Sources must include company name, year, excerpt text, and page number if available.
4. Always include any available revenue, profit, or growth figures mentioned in the text, 
but also clarify their scope (e.g., total revenue vs. segment revenue).

Question: {query}
Answer in JSON:

Rules:
1. If the exact figure requested (e.g., "total revenue 2022") is not directly available, 
   provide the closest available information (e.g., "Microsoft Cloud revenue was $91.2B") 
   and explicitly state that it is for a specific segment, not the total.
2. If multiple revenue/profit numbers are given, list them with clear labels.
3. Never ignore available numbers in the text, even if they are only for a segment.
4. If absolutely no relevant numerical figure is present, only then say "not available".


your answer template should be
{{
  "query": "Which company had the highest operating margin in 2023?",
  "answer": "Microsoft had the highest operating margin at 42.1% in 2023, followed by Google at 29.8% and NVIDIA at 29.6%.",
  "reasoning": "Retrieved operating margins for all three companies from their 2023 10-K filings and compared values.",
  "sub_queries": [
    "Microsoft operating margin 2023",
    "Google operating margin 2023",
    "NVIDIA operating margin 2023"
  ],
  "sources": [
    {{
      "company": "MSFT",
      "year": "2023",
      "excerpt": "Operating margin was 42.1%...",
      "page": 10
    }},
    {{
      "company": "GOOGL",
      "year": "2023",
      "excerpt": "Operating margin of 29.8%...",
      "page": 42
    }},
    {{
      "company": "NVDA",
      "year": "2023",
      "excerpt": "We achieved operating margin of 29.6%...",
      "page": 37
    }}
  ]
}}
"""
)

simple_prompt = PromptTemplate(
    input_variables=["company_data", "query"],
    template="""
You are a financial analysis agent.
Use the extracted text snippets {company_data} to:
1. Answer the query {query}.
2. If numbers for multiple years are present, calculate growth/percentage.
3. Return JSON with keys: query, answer, reasoning, sub_queries, sources.
4. Always include any available revenue, profit, or growth figures mentioned in the text, 
but also clarify their scope (e.g., total revenue vs. segment revenue).

Rules:
1. If the exact figure requested (e.g., "total revenue 2022") is not directly available, 
   provide the closest available information (e.g., "Microsoft Cloud revenue was $91.2B") 
   and explicitly state that it is for a specific segment, not the total.
2. If multiple revenue/profit numbers are given, list them with clear labels.
3. Never ignore available numbers in the text, even if they are only for a segment.
4. If absolutely no relevant numerical figure is present, only then say "not available".


your answer template should be
{{
  "query": "Which company had the highest operating margin in 2023?",
  "answer": "Microsoft had the highest operating margin at 42.1% in 2023, followed by Google at 29.8% and NVIDIA at 29.6%.",
  "reasoning": "Retrieved operating margins for all three companies from their 2023 10-K filings and compared values.",
  "sub_queries": [
    "Microsoft operating margin 2023",
    "Google operating margin 2023",
    "NVIDIA operating margin 2023"
  ],
  "sources": [
    {{
      "company": "MSFT",
      "year": "2023",
      "excerpt": "Operating margin was 42.1%...",
      "page": 10
    }},
    {{
      "company": "GOOGL",
      "year": "2023",
      "excerpt": "Operating margin of 29.8%...",
      "page": 42
    }},
    {{
      "company": "NVDA",
      "year": "2023",
      "excerpt": "We achieved operating margin of 29.6%...",
      "page": 37
    }}
  ]
}}
"""
)


# -----------------------------
# AGENT LOGIC
# -----------------------------
def decompose_query(query):
    comparative_keywords = ["compare", "highest", "change", "growth", "across all", "difference"]
    companies = ["Microsoft", "Google", "NVIDIA"]
    years = [str(y) for y in range(2022, 2025)]

    if any(word in query.lower() for word in comparative_keywords):
        sub_queries = []
        for c in companies:
            for y in years:
                sub_queries.append(f"{c} {query} {y}")
        return sub_queries, True
    else:
        return [query], False


def retrieve_and_summarize(vectorstore, sub_queries, is_comparative):
    company_data = []
    for sq in sub_queries:
        results = vectorstore.similarity_search(sq, k=3)
        for r in results:
            company_data.append({
                "filename": r.metadata.get("source", ""),
                "text": r.page_content[:500]
            })

    prompt_template = comparative_prompt if is_comparative else simple_prompt
    input_prompt = prompt_template.format(
        company_data=json.dumps(company_data),
        query=" / ".join(sub_queries)
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(input_prompt)

    return response.text

# -----------------------------
# MAIN (CLI LOOP)
# -----------------------------
def main():
    print(" Preloading FAISS vector store...")
    vectorstore = preload_vector_store()

    print("\n RAG Financial Analysis CLI (Gemini)")
    print("Type your query below (or type 'exit' to quit)\n")

    while True:
        query = input("🔎 Enter query: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting... ✅")
            break

        if not query:
            continue

        sub_queries, is_comp = decompose_query(query)
        response = retrieve_and_summarize(vectorstore, sub_queries, is_comp)
        print("\n Response:\n", response)
        print("-" * 80)


if __name__ == "__main__":
    main()