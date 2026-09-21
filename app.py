import os
import re
import tempfile
import requests
import streamlit as st
import gdown
from pypdf import PdfReader
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# Page Configuration
st.set_page_config(page_title="RAG Search with Groq & FAISS", page_icon="⚡", layout="wide")

st.title("⚡ Multi-Format RAG App: PDF, DOCX, TXT, MD & GDrive")

# Sidebar: API Keys & Model Selection
with st.sidebar:
    st.header("🔑 Configuration")
    
    # Check if key exists in Secrets
    secret_key = st.secrets.get("GROQ_API_KEY", "")
    if secret_key:
        st.success("✅ Groq API Key loaded from Secrets!")
        groq_api_key = secret_key
    else:
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Get your key at https://console.groq.com"
        )
    
    selected_model = st.selectbox(
        "Choose Groq Model",
        options=["openai/gpt-oss-120b", "qwen/qwen3.6-27b"],
        index=0
    )
    
    st.divider()
    st.markdown("### Processed Documents")

# Cache ML Models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedding_model = load_embedding_model()
reranker_model = load_reranker_model()

# Vector Store Class
class FAISSVectorStore:
    def __init__(self, model):
        self.model = model
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks_data = []

    def add_text(self, text: str, source_name: str, chunk_size: int = 500, chunk_overlap: int = 100):
        if not text.strip():
            return 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        raw_chunks = text_splitter.split_text(text)
        
        embeddings = []
        for idx, chunk_text in enumerate(raw_chunks):
            token_count = len(self.model.tokenizer.encode(chunk_text))
            chunk_obj = {
                "chunk_id": f"{source_name}_chunk_{idx}",
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "chunk_index": idx,
                    "token_count": token_count
                }
            }
            self.chunks_data.append(chunk_obj)
            embeddings.append(chunk_text)
            
        encoded_embeddings = self.model.encode(embeddings, convert_to_numpy=True)
        faiss.normalize_L2(encoded_embeddings)
        self.index.add(encoded_embeddings)
        return len(raw_chunks)

    def similarity_search(self, query: str, k: int = 10):
        if self.index.ntotal == 0:
            return []
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks_data) and idx != -1:
                results.append(self.chunks_data[idx])
        return results

if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore(embedding_model)

if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []

# File Text Extractors
def extract_text_from_file(file_obj, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    extracted_text = ""
    
    if ext == "pdf":
        reader = PdfReader(file_obj)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_text += f"\n--- Page {page_num + 1} ---\n" + text
    elif ext == "docx":
        doc = docx.Document(file_obj)
        extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif ext in ["txt", "md"]:
        if isinstance(file_obj, str):
            with open(file_obj, "r", encoding="utf-8") as f:
                extracted_text = f.read()
        else:
            extracted_text = file_obj.read().decode("utf-8")
            
    return extracted_text

def download_and_extract_gdrive(gdrive_url: str) -> str:
    # Handle Google Docs, Slides, Sheets exports
    if "docs.google.com/presentation" in gdrive_url:
        file_id = re.search(r'/d/([a-zA-Z0-9-_]+)', gdrive_url).group(1)
        export_url = f"https://docs.google.com/presentation/d/{file_id}/export/pdf"
        res = requests.get(export_url)
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(res.content)
        temp_pdf.close()
        text = extract_text_from_file(temp_pdf.name, "doc.pdf")
        os.remove(temp_pdf.name)
        return text
    elif "docs.google.com/document" in gdrive_url:
        file_id = re.search(r'/d/([a-zA-Z0-9-_]+)', gdrive_url).group(1)
        export_url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
        res = requests.get(export_url)
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(res.content)
        temp_pdf.close()
        text = extract_text_from_file(temp_pdf.name, "doc.pdf")
        os.remove(temp_pdf.name)
        return text
    else:
        # Direct file download using gdown
        match = re.search(r'[-_a-zA-Z0-9]{25,}', gdrive_url)
        file_id = match.group(0) if match else gdrive_url
        download_url = f'https://drive.google.com/uc?id={file_id}'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            gdown.download(download_url, temp_file.name, quiet=True)
            text = extract_text_from_file(temp_file.name, "gdrive_doc.pdf")
            return text
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

def rerank_chunks(query: str, retrieved_chunks: list, top_n: int = 3) -> list:
    if not retrieved_chunks:
        return []
    pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
    scores = reranker_model.predict(pairs)
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_n]]

# UI Layout: Ingestion
st.subheader("1. Document Ingestion")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT, DOCX, MD)", type=["pdf", "txt", "docx", "md"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting text & building vectors..."):
            text = extract_text_from_file(uploaded_file, uploaded_file.name)
            num_chunks = st.session_state.vector_store.add_text(text, source_name=uploaded_file.name)
            st.session_state.indexed_sources.append(f"📄 {uploaded_file.name} ({num_chunks} chunks)")
            st.success(f"Successfully processed {uploaded_file.name}!")

with col2:
    gdrive_link = st.text_input("Or enter Google Drive / Docs / Slides Link:")
    if gdrive_link and st.button("Download & Process GDrive Link"):
        with st.spinner("Downloading from Google Drive & indexing..."):
            try:
                text = download_and_extract_gdrive(gdrive_link)
                num_chunks = st.session_state.vector_store.add_text(text, source_name="GDrive_Doc")
                st.session_state.indexed_sources.append(f"☁️ Google Drive File ({num_chunks} chunks)")
                st.success("Successfully processed Google Drive file!")
            except Exception as e:
                st.error(f"Error processing GDrive link: {e}")

# Sidebar Status
with st.sidebar:
    if st.session_state.indexed_sources:
        for src in st.session_state.indexed_sources:
            st.write(src)
    else:
        st.info("No documents indexed yet.")

st.divider()

# UI Layout: Query
st.subheader("2. Ask Questions")
user_query = st.text_input("Enter your question:")

if st.button("Search & Answer", type="primary"):
    if not groq_api_key:
        st.error("Please provide a Groq API key.")
    elif st.session_state.vector_store.index.ntotal == 0:
        st.warning("Please index at least one document first.")
    elif not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching, Reranking & Fetching Answer from Groq..."):
            candidates = st.session_state.vector_store.similarity_search(user_query, k=10)
            reranked_chunks = rerank_chunks(user_query, candidates, top_n=3)
            
            context_str = ""
            for idx, c in enumerate(reranked_chunks, 1):
                context_str += f"\n[Document {idx} | Source: {c['metadata']['source']} | Tokens: {c['metadata']['token_count']}]\n{c['text']}\n"
            
            client = Groq(api_key=groq_api_key)
            system_prompt = "You are a helpful assistant. Use ONLY the provided context to answer the question."
            user_prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}\nAnswer:"
            
            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                
                st.markdown("### Answer")
                st.write(response.choices[0].message.content)
                
                with st.expander("🔍 View Top Reranked Chunks"):
                    for i, chunk in enumerate(reranked_chunks, 1):
                        st.markdown(f"**Chunk {i}** | Source: `{chunk['metadata']['source']}` | Tokens: `{chunk['metadata']['token_count']}`")
                        st.text(chunk["text"])
                        st.divider()
            except Exception as e:
                st.error(f"Groq API Error: {e}")
            
