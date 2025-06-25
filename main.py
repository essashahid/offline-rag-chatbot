from fastapi import FastAPI
from pydantic import BaseModel

from offline_rag import (
    extract_pdf_text,
    chunk_text,
    embed_chunks,
    build_index,
    retrieve,
    ask_hf_api,
)

app = FastAPI()


# Store resources in app state
@app.on_event("startup")
async def startup_event():
    PDF_PATH = "handbook.pdf"
    print("Loading PDF & building FAISS …")
    text = extract_pdf_text(PDF_PATH)
    chunks = chunk_text(text)
    embedder, embs = embed_chunks(chunks)
    faiss_idx = build_index(embs)

    app.state.chunks = chunks
    app.state.embedder = embedder
    app.state.faiss_idx = faiss_idx
    print("✅ Ready to answer requests!")


class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask(query: Query):
    chunks = app.state.chunks
    embedder = app.state.embedder
    faiss_idx = app.state.faiss_idx

    ctx = "\n".join(retrieve(query.question, embedder, faiss_idx, chunks))
    answer = ask_hf_api(ctx, query.question)
    return {"answer": answer}
