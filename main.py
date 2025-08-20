import os
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredXMLLoader
from langchain.schema import HumanMessage
from bs4 import BeautifulSoup
import pandas as pd
from langchain.docstore.document import Document

# -----------------------------
# Inside retrieve_and_summarize()
# -----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"  # Folder containing txt versions of 10-Ks
VECTOR_STORE_FILE = "faiss_index"

# -----------------------------
# LOAD DOCUMENTS AND CREATE VECTOR STORE
# -----------------------------
def load_documents(data_dir):
    """
    Load HTML, XHTML, and XML SEC filings.
    Extracts text + tables, converts tables into row-wise text for RAG.
    Returns a list of LangChain Document objects.
    """
    all_docs = []

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue

        chunks = []

        # HTML / XHTML files
        if filename.lower().endswith((".html", ".htm", ".xhtml")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")

            # Extract main text
            text = soup.get_text(separator="\n")
            if text.strip():
                chunks.append(text)

            # Extract all tables
            try:
                tables = pd.read_html(str(soup))
                for table in tables:
                    for _, row in table.iterrows():
                        row_text = " | ".join([f"{c}: {row[c]}" for c in table.columns])
                        chunks.append(row_text)
            except ValueError:
                # No tables found
                pass

        # XML files (e.g., XBRL)
        elif filename.lower().endswith(".xml"):
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "xml")
            text = soup.get_text(separator="\n")
            if text.strip():
                chunks.append(text)

        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        # Convert chunks into LangChain Documents
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"source": filename})
            all_docs.append(doc)

        print(f"Loaded {len(chunks)} chunks from {filename}")

    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

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
2. Returning a JSON with keys: query, answer, reasoning, sub_queries, sources each should seprate by new line.
3. Sources must include company name, year, excerpt text, and page number if available.

Question: {query}
Answer in JSON:
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
"""
)

# -----------------------------
# AGENT LOGIC
# -----------------------------
def decompose_query(query):
    """Detect if comparative query and generate sub-queries."""
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

def retrieve_and_summarize(vectorstore, sub_queries, llm, is_comparative):
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
     # Proper invocation with HumanMessage
    answer = llm.invoke([HumanMessage(content=input_prompt)]).content
    return answer

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading documents and creating vector store...")
    docs = load_documents(DATA_DIR)
    vectorstore = create_vector_store(docs)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    
    for q in test_queries:
        sub_queries, is_comp = decompose_query(q)
        print("\nQuery:", q)
        response = retrieve_and_summarize(vectorstore, sub_queries, llm, is_comp)
        print("Response:\n", response)

if __name__ == "__main__":
    main()
