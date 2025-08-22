import os
import json
import pdfplumber
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"          # folder with your 9 PDFs
CHROMA_DB_DIR = "./chroma_store"

# -----------------------------
# INIT GEMINI
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# LOAD PDFs (text + tables)
# -----------------------------
def load_pdfs(data_dir):
    all_docs = {}
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path) or not filename.lower().endswith(".pdf"):
            continue

        docs = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # --- Extract text ---
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1, "type": "text"}
                            )
                        )

                    # --- Extract tables ---
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if table:
                            table_text = "\n".join(
                                [" | ".join([str(cell) if cell else "" for cell in row]) for row in table]
                            )
                            if table_text.strip():
                                docs.append(
                                    Document(
                                        page_content=table_text,
                                        metadata={"source": filename, "page": i + 1, "type": "table", "table_index": t_idx}
                                    )
                                )
                                

            print(f"✅ Parsed {filename} ({len(pdf.pages)} pages)")
            all_docs[filename] = docs

        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")

    return all_docs

# -----------------------------
# CREATE / LOAD CHROMA STORES
# -----------------------------
def create_chroma_store(company_name, docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma(
        collection_name=f"financial_reports_{company_name}",
        embedding_function=embeddings,
        persist_directory=os.path.join(CHROMA_DB_DIR, company_name)  
    )
    vectorstore.add_documents(chunks)
    return vectorstore

def preload_all_stores():
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)

    docs_dict = load_pdfs(DATA_DIR)
    stores = {}

    for company, docs in docs_dict.items():
        print(f"📊 Creating/loading vector store for {company}...")

        persist_path = os.path.join(CHROMA_DB_DIR, company)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=GEMINI_API_KEY
        )

        # ✅ If DB already exists, just load it
        if os.path.exists(persist_path) and os.listdir(persist_path):
            vectorstore = Chroma(
                collection_name=f"financial_reports_{company}",
                embedding_function=embeddings,
                persist_directory=persist_path
            )
        else:
            # ✅ Otherwise, build it once
            vectorstore = create_chroma_store(company, docs)

        stores[company] = vectorstore

    return stores

# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
analysis_prompt = PromptTemplate(
    input_variables=["company_data", "query"],
    template="""
You are a financial analysis assistant. 
Use ONLY the provided company data below:
{company_data}

Question: {query}

Instructions:
- Always provide an answer, even if only partial data is available.
- If some companies are missing values, estimate or clarify based on available context.
- Always return JSON strictly in this format:

{{
  "query": "<repeat the input query>",
  "answer": "<direct clear answer, comparison if needed>",
  "reasoning": "<explain how you derived answer step by step>",
  "sub_queries": [
      "<break query into smaller per-company sub-queries>"
  ],
  "sources": [
    {{
      "company": "<company ticker or name>",
      "year": "<year if available>",
      "excerpt": "<short excerpt from context>",
      "page": <page number if available>
    }}
  ]
}}

Rules:
- Do NOT say "answer not available".
- Do NOT invent fake companies; only use those from the context.
- If data is missing, still provide best effort answer from context.
- Keep excerpts short (1-2 sentences max).
"""
)

# -----------------------------
# RAG PIPELINE
# -----------------------------
def retrieve_and_answer(stores, query):
    all_results = []

    # search across all companies (comparison possible)
    for company, store in stores.items():
        results = store.similarity_search(query, k=5)
        for r in results:
            all_results.append({
                "company": company,
                "source": r.metadata.get("source", ""),
                "page": r.metadata.get("page", ""),
                "text": r.page_content[:400]
            })

    # Build prompt
    input_prompt = analysis_prompt.format(
        company_data=json.dumps(all_results, indent=2),
        query=query
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(input_prompt)

    return response.text

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("📊 Preloading all Chroma stores...")
    stores = preload_all_stores()

    print("\n💬 Financial RAG CLI (Chroma + Gemini, multi-company ready)")
    while True:
        query = input("🔎 Enter query (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        answer = retrieve_and_answer(stores, query)
        print("\n📌 Response:\n", answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
