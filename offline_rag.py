import os, sys, json, time, requests
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

HF_TOKEN = os.getenv("HF_TOKEN")

# --- Config ---
# HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


# ---------------------------------------------------------------------
# 1. Extract text from PDF
def extract_pdf_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return ""

    with pdfplumber.open(pdf_path) as pdf:
        print(f"📄 PDF has {len(pdf.pages)} pages.")
        pages = []
        for i, p in enumerate(pdf.pages):
            txt = p.extract_text()
            if txt:
                print(f"✅ page {i+1}")
                pages.append(txt)
            else:
                print(f"⚠️ empty page {i+1}")
    return "\n".join(pages)


# ---------------------------------------------------------------------
# 2. Chunk text
def chunk_text(text: str, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    print(f"✂️ {len(chunks)} chunks created")
    return chunks


# ---------------------------------------------------------------------
# 3. Embeddings
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    print(f"🔍 loading embedder: {model_name}")
    model = SentenceTransformer(model_name)
    print("🧠 encoding chunks…")
    emb = model.encode(chunks, show_progress_bar=True)
    print(f"✅ embeddings shape {np.array(emb).shape}")
    return model, emb


# ---------------------------------------------------------------------
# 4. Build FAISS
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"💾 FAISS index ready ({embeddings.shape[0]} vectors)")
    return index


# ---------------------------------------------------------------------
# 5. Retrieve chunks
def retrieve(query, model, index, chunks, k=3):
    print(f"🔎 retrieving top {k} for: {query}")
    q_emb = model.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return [chunks[i] for i in idx[0]]


# ---------------------------------------------------------------------
# 6. Remote LLM call
def ask_hf_api(context: str, question: str, max_new_tokens=200) -> str:
    if not HF_TOKEN:
        sys.exit("❌ Set HF_TOKEN env var first!")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    prompt = (
        "Answer the question using only the context.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2},
    }

    start = time.time()
    r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    resp = r.json()
    latency = time.time() - start

    # HF returns a list on success
    txt = resp[0]["generated_text"] if isinstance(resp, list) else str(resp)
    print(f"🌐 HF latency: {latency:.1f}s")
    return txt.strip()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    PDF_PATH = "handbook.pdf"

    print("🔍 extracting PDF…")
    full_text = extract_pdf_text(PDF_PATH)
    if not full_text.strip():
        sys.exit("❌ no text extracted – exiting")

    chunks = chunk_text(full_text)
    embedder, embs = embed_chunks(chunks)
    faiss_idx = build_index(np.array(embs))

    # interactive loop
    print("\n✅ Ready! (type 'exit' to quit)")
    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        ctx = "\n".join(retrieve(q, embedder, faiss_idx, chunks))
        print("🤖 asking Hugging Face API…")
        answer = ask_hf_api(ctx, q)
        print("\n💬 Answer:\n", answer)
