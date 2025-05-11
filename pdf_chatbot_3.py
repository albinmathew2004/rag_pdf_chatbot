import os
import uuid
import json
import time
import hashlib
import requests
import streamlit as st
from io import BytesIO  # ✅ Needed to fix the PDF seek error

# ✅ Must come first in Streamlit
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()
GENIE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
genai.configure(api_key=GENIE_API_KEY)
CHROMADB_CLIENT = chromadb.Client()

# Cache the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

MODEL = load_model()

# Get or create ChromaDB collection
collection = CHROMADB_CLIENT.get_or_create_collection("pdf_chunks")

# Generate a hash for uploaded files
def hash_file(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# ✅ Fixed: Use BytesIO to avoid seek error
@st.cache_data(show_spinner=False)
def extract_pdf_components_bytes(file_bytes):
    elements = partition_pdf(file=BytesIO(file_bytes), strategy="fast")  # ✅ Use fast strategy to avoid layout model
    text_chunks, table_chunks = [], []
    for el in elements:
        if el.category == "NarrativeText":
            text_chunks.append(el.text)
        elif el.category == "Table":
            table_chunks.append(el.text)
    return text_chunks, table_chunks

# Embed chunks
def embed_chunks(doc_id, chunks, chunk_type):
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{chunk_type}_{i}"
        embedding = MODEL.encode(chunk)
        collection.add(documents=[chunk], ids=[chunk_id], metadatas=[{
            "doc_id": doc_id,
            "type": chunk_type,
            "chunk_index": i
        }])

# Store document locally
def store_full_doc(doc_id, texts, tables):
    os.makedirs("data", exist_ok=True)
    full_doc_path = "data/full_docs.json"
    data = {}
    if os.path.exists(full_doc_path):
        try:
            with open(full_doc_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data[doc_id] = {"texts": texts, "tables": tables}
    with open(full_doc_path, "w") as f:
        json.dump(data, f, indent=2)

# Groq summarization
def call_llama_api(text):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": f"Summarize this:\n{text}"}],
        "temperature": 0.3
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError:
            if r.status_code == 429:
                time.sleep(3)
            else:
                break
        except requests.exceptions.RequestException as e:
            return f"Error summarizing text: {str(e)}"
    return "❌ Failed to summarize after retries."

# Query pipeline using top-K semantic search
def query_pipeline(query, top_k=10, max_tokens=3500):
    query_emb = MODEL.encode(query)
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    if not results.get("ids") or not results["ids"][0]:
        return "❌ No relevant documents found."

    top_chunks = results["documents"][0]
    combined_text = "\n\n".join(top_chunks)
    if len(combined_text) > max_tokens:
        combined_text = combined_text[:max_tokens]

    prompt = (
        f"You are a financial expert chatbot. Based on the following extracted document content, "
        f"give a detailed and comprehensive answer to the question.\n\n"
        f"---\n{combined_text}\n---\n\n"
        f"Question: {query}\n\n"
        f"Be specific and elaborate if the information is present. If partial information is found, state that clearly."
    )

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"❌ Error generating answer: {str(e)}"

# Sidebar and File Upload
st.sidebar.title("📄 PDF Chatbot")
st.sidebar.markdown("Upload a PDF and ask detailed questions using semantic search + LLaMA.")
uploaded_file = st.sidebar.file_uploader("Upload PDF File", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hash_file(file_bytes)

    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = {}

    if file_hash not in st.session_state["processed_files"]:
        with st.spinner("🔍 Extracting content..."):
            text_chunks, table_chunks = extract_pdf_components_bytes(file_bytes)

        doc_id = str(uuid.uuid4())
        with st.spinner("💾 Embedding & Storing..."):
            embed_chunks(doc_id, text_chunks, "text")
            embed_chunks(doc_id, table_chunks, "table")
            store_full_doc(doc_id, text_chunks, table_chunks)

        st.session_state["processed_files"][file_hash] = doc_id
        st.session_state["doc_id"] = doc_id
        st.sidebar.success(f"✅ Document processed!\nDoc ID: {doc_id}")
    else:
        st.session_state["doc_id"] = st.session_state["processed_files"][file_hash]
        st.sidebar.info(f"📁 Document already processed.\nDoc ID: {st.session_state['doc_id']}")

# Track chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Main Input and Output
st.header("Ask a Question About Your Document")
query = st.text_input("Type your question below:")

if query:
    with st.spinner("🤖 Thinking..."):
        response = query_pipeline(query)

    # Store the conversation in chat history
    st.session_state["chat_history"].append({"question": query, "answer": response})

    st.write(response)

# Display chat history
if st.session_state["chat_history"]:
    st.subheader("Full Chat History:")
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.markdown("---")

st.markdown("---")
st.markdown("""
### About
This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded documents.
Built with Streamlit, SentenceTransformers, ChromaDB, and Groq’s LLaMA models.
""")
