from __future__ import annotations

import os
import math
import threading
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Optional: Pathway imports (installed per README). We keep usage light for hackathon.
try:
    import pathway as pw  # noqa: F401
except Exception:  # pragma: no cover - optional at runtime
    pw = None

try:
    # Prefer a real embedder if available
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class IngestDoc(BaseModel):
    id: str
    title: Optional[str] = None
    text: str
    source: Optional[str] = None
    published_at: Optional[str] = None


class IngestRequest(BaseModel):
    ticker: str
    docs: List[IngestDoc]


class IngestResponse(BaseModel):
    upserts: int
    duplicates: int
    total: int


class QueryRequest(BaseModel):
    ticker: str
    question: str
    k: int = Field(default=5, ge=1, le=50)


class ChunkOut(BaseModel):
    id: str
    title: Optional[str] = None
    text: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    chunks: List[ChunkOut]
    rationale: str = "pathway"
    used: str = "pathway"


def _normalize(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    sa = 0.0
    sb = 0.0
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
        sa += x * x
        sb += y * y
    denom = (math.sqrt(sa) or 1.0) * (math.sqrt(sb) or 1.0)
    return dot / denom


def _tokenize(text: str) -> List[str]:
    return [t for t in ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in (text or '').lower()).split() if len(t) > 1]


def _chunk_text(text: str, max_tokens: int = 300, overlap: int = 60) -> List[str]:
    tokens = _tokenize(text)
    chunks: List[str] = []
    start = 0
    idx = 0
    max_tokens = max(64, min(512, max_tokens))
    overlap = max(0, min(max_tokens // 2, overlap))
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        chunk = ' '.join(window).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_tokens >= len(tokens):
            break
        start += (max_tokens - overlap)
        idx += 1
    return chunks


class HashedEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _hash32(s: str) -> int:
        h = 2166136261
        for ch in s:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _token_vec(self, token: str) -> List[float]:
        seed = self._hash32(token)
        v = [0.0] * self.dim
        for i in range(self.dim):
            seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
            x = ((seed >> 9) & 0x7FFFFF) / float(0x7FFFFF)
            v[i] = (x * 2.0 - 1.0)
        return v

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            tokens = _tokenize(text)
            vec = [0.0] * self.dim
            for t in tokens:
                tv = self._token_vec(t)
                for i in range(self.dim):
                    vec[i] += tv[i]
            out.append(_normalize(vec))
        return out


class SBertEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer not available")
        self.model = SentenceTransformer(model_name)
        self.lock = threading.Lock()

    def embed(self, texts: List[str]) -> List[List[float]]:
        with self.lock:
            vecs = self.model.encode(texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def build_embedder() -> Any:
    choice = (os.getenv("EMBEDDER") or "hash").lower()
    if choice == "sbert":
        try:
            return SBertEmbedder()
        except Exception:
            return HashedEmbedder()
    return HashedEmbedder()


class VectorIndex:
    def __init__(self, embedder: Any):
        # ticker -> { chunk_id -> (vector, payload) }
        self.by_ticker: Dict[str, Dict[str, Tuple[List[float], Dict[str, Any]]]] = {}
        self.embedder = embedder

    def upsert_docs(self, ticker: str, docs: List[IngestDoc]) -> Tuple[int, int, int]:
        bucket = self.by_ticker.setdefault(ticker.upper(), {})
        upserts = 0
        duplicates = 0
        # Build chunk list with (id, text, meta)
        chunk_records: List[Tuple[str, str, Dict[str, Any]]] = []
        for d in docs:
            base_id = d.id
            pieces = _chunk_text(d.text or "")
            for i, chunk_text in enumerate(pieces):
                cid = f"{base_id}#{i}"
                meta = {
                    "id": cid,
                    "title": d.title or (d.text[:140] if d.text else None),
                    "text": chunk_text,
                    "source": d.source,
                    "published_at": d.published_at,
                }
                chunk_records.append((cid, chunk_text, meta))
        # Embed in batches
        vectors = self.embedder.embed([t for (_, t, _) in chunk_records]) if chunk_records else []
        for (cid, _, meta), vec in zip(chunk_records, vectors):
            if cid in bucket:
                duplicates += 1
            else:
                upserts += 1
            bucket[cid] = (vec, meta)
        total = len(bucket)
        return upserts, duplicates, total

    def query(self, ticker: str, question: str, k: int) -> List[ChunkOut]:
        bucket = self.by_ticker.get(ticker.upper())
        if not bucket:
            return []
        qv = self.embedder.embed([question])[0]
        scored: List[Tuple[str, float]] = []
        for cid, (vec, meta) in bucket.items():
            s = _cosine(qv, vec)
            scored.append((cid, float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        out: List[ChunkOut] = []
        for cid, score in top:
            meta = bucket[cid][1]
            out.append(ChunkOut(id=cid, title=meta.get("title"), text=meta.get("text", ""), source=meta.get("source"), published_at=meta.get("published_at"), score=score))
        return out


app = FastAPI(title="Pathway RAG Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_EMBEDDER = build_embedder()
_INDEX = VectorIndex(_EMBEDDER)


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.ticker or not req.docs:
        raise HTTPException(status_code=400, detail="ticker and docs required")
    up, dup, tot = _INDEX.upsert_docs(req.ticker, req.docs)
    return IngestResponse(upserts=up, duplicates=dup, total=tot)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.ticker or not req.question:
        raise HTTPException(status_code=400, detail="ticker and question required")
    chunks = _INDEX.query(req.ticker, req.question, req.k)
    return QueryResponse(chunks=chunks)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)


