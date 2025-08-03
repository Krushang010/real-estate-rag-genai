import os
import uuid
import time
from typing import List, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------- ENVIRONMENT -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (MyApp/1.0)")
os.environ["USER_AGENT"] = USER_AGENT

if not GROQ_API_KEY:
    raise EnvironmentError("🚫 GROQ_API_KEY is not set in your .env file.")

# ------------------------- GLOBAL MODELS -------------------------
collections = {}
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------------- LLM INITIALIZATION -------------------------
def init_llm(model_name: str = "llama3-8b-8192", temperature=0.9):
    return ChatGroq(api_key=GROQ_API_KEY, model_name=model_name, temperature=temperature)

# ------------------------- VECTORSTORE SETUP -------------------------
def reset_collection(persist_directory: str, collection_name: str):
    full_path = os.path.join(persist_directory, collection_name)
    if os.path.exists(full_path):
        import shutil
        shutil.rmtree(full_path)
        print(f"🔁 Reset vectorstore at: {full_path}")

def fetch_clean_url(url: str) -> Document:
    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def load_urls_parallel(urls: List[str], max_workers: int = 6) -> List[Document]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        docs = list(executor.map(fetch_clean_url, urls))
    return [doc for doc in docs if doc]

def load_and_prepare_docs(
    urls: List[str],
    collection_name: str = "default_collection",
    persist_directory: str = "chroma_db",
    reset: bool = False
) -> Chroma:
    if reset:
        reset_collection(persist_directory, collection_name)

    if collection_name in collections and not reset:
        return collections[collection_name]

    docs = load_urls_parallel(urls)

    if not docs or all(not doc.page_content.strip() for doc in docs):
        raise RuntimeError("⚠️ All provided URLs returned empty or invalid content.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    # ✅ TEMPORARY limit chunks to avoid memory crash
    chunks = chunks[:50]  # <-- You can increase this slowly (20, 50, etc.)

    print(f"🔢 Embedding {len(chunks)} chunks...")

    for doc in chunks:
        doc.metadata["uuid"] = str(uuid.uuid4())

    # ✅ RAM-safe embedding (chunk by chunk if needed)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=None,  # disables persistence
        is_persistent=False  # ✅ ensures in-memory mode
    )

    collections[collection_name] = vectorstore
    return vectorstore


# ------------------------- QA CHAIN -------------------------
def get_qa_chain(
    vectorstore: Chroma,
    llm=None,
    return_sources: bool = False
) -> Tuple[RetrievalQA, ChatGroq]:
    if llm is None:
        llm = init_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=return_sources
    )
    return qa_chain, llm

# ------------------------- UTILITIES -------------------------
def summarize_text(text, max_chars=2000):
    cleaned = text.strip()
    if not cleaned:
        return ""
    paragraphs = cleaned.split("\n")
    result = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(result) + len(para) + 1 > max_chars:
            break
        result += para + "\n"
    return result.strip()

def generate_prompt(question: str, combined_text: str) -> str:
    question_lower = question.lower()

    if "challenge" in question_lower or "barrier" in question_lower:
        instruction = (
            "Extract a bullet-point list of key *regulatory and economic challenges* "
            "slowing down real estate growth in India. Avoid general advice or trends.\n\n"
        )
    elif "trend" in question_lower or "update" in question_lower:
        instruction = (
            "Summarize the following into a concise, bullet-point list of *real estate trends* "
            "in India, based on the latest available information.\n\n"
        )
    elif "impact" in question_lower or "effect" in question_lower:
        instruction = (
            "Describe the *impact* or lack thereof of the following content. If no clear impact is found, list the most relevant real estate insights instead.\n\n"
        )
    elif "opportunity" in question_lower or "growth area" in question_lower:
        instruction = (
            "Extract a bullet-point list of *emerging opportunities or growth areas* in the Indian real estate market.\n\n"
        )
    else:
        instruction = (
            "Summarize the relevant insights from the content below as a concise bullet-point list.\n\n"
        )

    return instruction + combined_text

def safe_invoke(llm, prompt, retries=3, delay=65):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                print(f"⏳ Rate limited. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
    raise RuntimeError("❌ Exceeded retry attempts due to persistent rate limiting.")

# ------------------------- MAIN RAG PIPELINE -------------------------
def run_rag_pipeline(question: str, qa_chain, llm) -> Tuple[str, List[str]]:
    response = qa_chain.invoke({"query": question})
    answer = response['result']
    source_docs = response.get("source_documents", [])

    combined_text = "\n\n".join(
        [summarize_text(doc.page_content, max_chars=2000) for doc in source_docs]
    )

    prompt = generate_prompt(question, combined_text)
    refined_summary = safe_invoke(llm, prompt)
    summary_text = refined_summary.content.strip()

    if not summary_text or "no specific" in summary_text.lower():
        fallback_prompt = "Answer the question using general economic knowledge: " + question
        fallback_response = safe_invoke(llm, fallback_prompt)
        summary_text = fallback_response.content.strip()

    sources = list({doc.metadata.get("source") for doc in source_docs})
    return summary_text, sources

# ------------------------- CLI EXECUTION -------------------------
if __name__ == "__main__":
    urls = [
        "https://realty.economictimes.indiatimes.com/news/residential",
        "https://timesofindia.indiatimes.com/city/surat",
        "https://propertywire.com/",
        "https://www.zillow.com/research/",
        "https://www.financialexpress.com/market/",
        "https://www.forbes.com/real-estate/",
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
        "https://www.moneycontrol.com/news/business/real-estate/"
    ]

    print("⏳ Loading and embedding URLs...")
    vectorstore = load_and_prepare_docs(
        urls=urls,
        collection_name="real_estate_collection",
        persist_directory=None,
        reset=False
    )

    qa_chain, llm = get_qa_chain(vectorstore, return_sources=True)

    question = "Find the trend of real estate market India?"
    print("❓ Question:", question)

    try:
        summary, sources = run_rag_pipeline(question, qa_chain, llm)
        print("\n📌 Refined Summary:\n", summary)
        print("\n🔗 Sources:")
        for src in sources:
            print("•", src)
    except Exception as e:
        print(f"❌ Error: {e}")
