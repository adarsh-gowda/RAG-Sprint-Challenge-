import os
import json
import pdfplumber
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Use new Chroma package
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data1"          # folder with your 10-K PDFs
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
    docs = []
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

        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")

    print(f"\n📄 Total documents loaded: {len(docs)}")
    return docs

# -----------------------------
# CREATE / LOAD CHROMA
# -----------------------------
def create_chroma_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma(
        collection_name="financial_reports",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return vectorstore

def preload_chroma_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GEMINI_API_KEY
    )
    if os.path.exists(CHROMA_DB_DIR):
        print("✅ Loading existing Chroma store...")
        return Chroma(
            collection_name="financial_reports",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
    else:
        print("⚠️ No Chroma DB found. Creating new one...")
        docs = load_pdfs(DATA_DIR)
        return create_chroma_store(docs)

# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
analysis_prompt = PromptTemplate(
    input_variables=["company_data", "query"],
    template="""
You are a financial analysis assistant.
Use the extracted financial data: {company_data}
Answer the question: {query}

Rules:
1. Always provide an answer, even if only partial or segment-level data is available.
2. Clearly indicate if numbers are segment-specific and not totals.
3. If totals or full data are missing, create sub-queries to retrieve missing pieces from the data.
4. Explain your reasoning and highlight any assumptions made.
5. Include sources with company, year, excerpt, and page if available.

Return JSON with keys:
- query: original query
- sub_queries: list of sub-queries you used to infer or fill gaps
- answer: your final answer
- reasoning: detailed explanation of your approach
- sources
"""
)

# -----------------------------
# RAG PIPELINE
# -----------------------------
def retrieve_and_answer(vectorstore, query):
    results = vectorstore.similarity_search(query, k=10)

    company_data = []
    for r in results:
        company_data.append({
            "source": r.metadata.get("source", ""),
            "page": r.metadata.get("page", ""),
            "text": r.page_content[:400]
        })

    input_prompt = analysis_prompt.format(
        company_data=json.dumps(company_data),
        query=query
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(input_prompt)

    return response.text

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("📊 Preloading Chroma vector store...")
    vectorstore = preload_chroma_store()

    print("\n💬 Financial RAG CLI (Chroma + Gemini)")
    while True:
        query = input("🔎 Enter query (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        answer = retrieve_and_answer(vectorstore, query)
        print("\n📌 Response:\n", answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
