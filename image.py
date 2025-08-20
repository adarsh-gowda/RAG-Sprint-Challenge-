# rag_financial_hybrid.py
import os
import io
import json
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

# ---------- Google / LangChain ----------
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# ---------- PDF / Images ----------
import pdfplumber
from PIL import Image

# =========================
# CONFIG
# =========================
DATA_DIR = "data1"                 # <--- your 9 PDFs here
VECTOR_STORE_FILE = "faiss_index_hybrid"
EMBEDDING_MODEL = "models/embedding-001"
TEXT_MODEL = "gemini-2.0-flash"   # for synthesis
VISION_MODEL = "gemini-1.5-flash" # OCR fallback on page images
OCR_IF_EMPTY_ONLY = True          # True => only OCR if page text+tables are empty

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in environment")
genai.configure(api_key=GEMINI_API_KEY)

# =========================
# PROMPTS
# =========================
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

# =========================
# PDF → TEXT/TABLES (+ Vision OCR fallback)
# =========================
def page_to_pil(page, resolution=300) -> Image.Image:
    """Render a pdfplumber page to a PIL Image."""
    # pdfplumber PageImage.original is a PIL Image
    return page.to_image(resolution=resolution).original

def stringify_table(table: List[List[Any]]) -> str:
    """Convert a 2D table list into a readable pipe-delimited string."""
    lines = []
    for row in table:
        if not row: 
            continue
        cells = [(str(c).strip() if c is not None else "") for c in row]
        # skip empty rows
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)

def ocr_with_gemini(pil_img: Image.Image) -> str:
    """Use Gemini Vision to OCR/understand a page image and return structured text."""
    try:
        model = genai.GenerativeModel(VISION_MODEL)
        resp = model.generate_content(
            [pil_img, 
             "Extract all readable text and any tabular data in a structured, line-separated format. "
             "If a table is present, emit rows separated by newlines and cells separated by ' | '."]
        )
        return resp.text or ""
    except Exception as e:
        print(f"⚠️ Vision OCR error: {e}")
        return ""

def load_documents(data_dir: str) -> List[Document]:
    all_docs: List[Document] = []

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not files:
        print(f"No PDFs found in {data_dir}")
        return all_docs

    for filename in files:
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue

        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    added_any = False

                    # --- 1) TEXT ---
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        text = ""
                        print(f"⚠️ extract_text error on {filename} p{page_num}: {e}")

                    if text and text.strip():
                        all_docs.append(Document(
                            page_content=text,
                            metadata={"source": filename, "page": page_num, "type": "text"}
                        ))
                        added_any = True

                    # --- 2) TABLES ---
                    try:
                        tables = page.extract_tables()
                    except Exception as e:
                        tables = []
                        print(f"⚠️ extract_tables error on {filename} p{page_num}: {e}")

                    if tables:
                        for t_idx, table in enumerate(tables):
                            table_text = stringify_table(table)
                            if table_text.strip():
                                all_docs.append(Document(
                                    page_content=table_text,
                                    metadata={
                                        "source": filename,
                                        "page": page_num,
                                        "type": "table",
                                        "table_index": t_idx
                                    }
                                ))
                                added_any = True

                    # --- 3) Vision OCR fallback (only if nothing added) ---
                    if (not added_any) and OCR_IF_EMPTY_ONLY:
                        try:
                            pil_img = page_to_pil(page, resolution=300)
                            ocr_text = ocr_with_gemini(pil_img)
                            if ocr_text.strip():
                                all_docs.append(Document(
                                    page_content=ocr_text,
                                    metadata={"source": filename, "page": page_num, "type": "ocr"}
                                ))
                                added_any = True
                        except Exception as e:
                            print(f"⚠️ OCR fallback error on {filename} p{page_num}: {e}")

            print(f"✅ Loaded {len(pdf.pages)} pages (text/tables{' + OCR' if OCR_IF_EMPTY_ONLY else ''}) from {filename}")

        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")

    print(f"\n📄 Total documents loaded: {len(all_docs)}")
    return all_docs

# =========================
# VECTOR STORE
# =========================
def create_vector_store(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_FILE)
    return vectorstore

def preload_vector_store():
    if os.path.exists(VECTOR_STORE_FILE):
        print("✅ Vector store already exists. Loading...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=GEMINI_API_KEY
        )
        return FAISS.load_local(VECTOR_STORE_FILE, embeddings, allow_dangerous_deserialization=True)
    else:
        print("⚠️ Vector store not found. Creating new one from data folder...")
        docs = load_documents(DATA_DIR)
        return create_vector_store(docs)

# =========================
# AGENT LOGIC
# =========================
def decompose_query(query: str) -> Tuple[List[str], bool]:
    comparative_keywords = ["compare", "highest", "change", "growth", "across all", "difference", "versus", "vs"]
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

def retrieve_and_summarize(vectorstore, sub_queries: List[str], is_comparative: bool) -> str:
    company_data = []
    for sq in sub_queries:
        results = vectorstore.similarity_search(sq, k=3)
        for r in results:
            company_data.append({
                "filename": r.metadata.get("source", ""),
                "page": r.metadata.get("page"),
                "type": r.metadata.get("type"),
                "text": r.page_content[:800]
            })

    prompt_template = comparative_prompt if is_comparative else simple_prompt
    input_prompt = prompt_template.format(
        company_data=json.dumps(company_data),
        query=" / ".join(sub_queries)
    )

    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(input_prompt)
    return response.text

# =========================
# CLI
# =========================
def main():
    print(" Preloading FAISS vector store...")
    vectorstore = preload_vector_store()

    print("\n RAG Financial Analysis CLI (Gemini)")
    print("Type your query below (or type 'exit' to quit)\n")

    while True:
        try:
            query = input("🔎 Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting... ✅")
            break

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
