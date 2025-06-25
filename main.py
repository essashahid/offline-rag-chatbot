# main.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from offline_rag import (  # your helper module
    extract_pdf_text,
    chunk_text,
    embed_chunks,
    build_index,
    retrieve,
    ask_hf_api,
)

app = FastAPI()

# an asyncio Event to signal when everything is ready
ready: asyncio.Event = asyncio.Event()


async def _warm_up() -> None:
    """
    Heavy, blocking work runs here — but in a thread,
    so the event loop (and therefore Uvicorn) keeps going.
    """
    PDF_PATH = "handbook.pdf"
    print("⬇️  Loading PDF & building FAISS …")

    # Heavy work 👉 run in a worker thread
    def _build():
        text = extract_pdf_text(PDF_PATH)
        chunks = chunk_text(text)
        embedder, embs = embed_chunks(chunks)
        faiss_idx = build_index(embs)
        return chunks, embedder, faiss_idx

    chunks, embedder, faiss_idx = await asyncio.to_thread(_build)

    # cache in app.state so every request can reuse them
    app.state.chunks = chunks
    app.state.embedder = embedder
    app.state.faiss_idx = faiss_idx

    ready.set()  # tell the world we’re ready
    print("✅  FAISS ready — API can now answer questions!")


@app.on_event("startup")
async def startup_event() -> None:
    """
    Kick off warm-up in the background and return immediately.
    This lets Uvicorn bind to the port almost instantly.
    """
    asyncio.create_task(_warm_up())


class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask(query: Query):
    # wait until the background warm-up finished
    if not ready.is_set():
        raise HTTPException(
            status_code=503, detail="Index warming-up. Try again in ~30 s."
        )

    chunks = app.state.chunks
    embedder = app.state.embedder
    faiss_idx = app.state.faiss_idx

    ctx = "\n".join(retrieve(query.question, embedder, faiss_idx, chunks))
    answer = ask_hf_api(ctx, query.question)
    return {"answer": answer}
