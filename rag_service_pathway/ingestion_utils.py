# ingestion_utils.py
"""
Robust ingestion utilities for the Pathway RAG microservice.

Features:
- fetch_text_from_url(): browser-like GET + variants + retries + proxy fallback (r.jina.ai)
- chunk_text(): simple character-based chunking
- sentence-transformers local embedder + normalization
- FAISS Index (IndexFlatIP) for cosine via normalized vectors
- persistent metadata (meta.pkl) and index (/app/data/vectors.index)
- high-level ingestion: ingest_url_to_index(url)
- query helper: query_by_text(query, top_k)

Save to: /app/ingestion_utils.py (in container). In this project, place at:
C:\college\snaptocook\rag_service_pathway\ingestion_utils.py
"""

from __future__ import annotations
import os
import time
import pickle
import requests
from readability import Document
from bs4 import BeautifulSoup
from typing import List, Tuple
import numpy as np
import faiss
import logging

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion_utils")

# persistent paths inside the container
DATA_DIR = "/app/data"
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
INDEX_PATH = os.path.join(DATA_DIR, "vectors.index")

# embedding model name
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# singletons
_embedder = None
_index = None
_dim = None

# ensure data dir
os.makedirs(DATA_DIR, exist_ok=True)


# -------------------------
# Fetch & extraction logic
# -------------------------
def fetch_text_from_url(url: str, timeout: int = 15) -> Tuple[str, str]:
    """
    Robust fetch + extract:
      - tries variants (with/without trailing slash, toggle www)
      - browser-like headers, follows redirects
      - retries primary attempts
      - if primary attempts fail or extracted text is too short, falls back to r.jina.ai proxy with retries
    Returns: (title, text)
    Raises: RuntimeError if all attempts fail with summarized errors.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    def parse_html(text_html: str, fallback_title: str) -> Tuple[str, str]:
        doc = Document(text_html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        title = doc.short_title() or (soup.title.string if soup.title else fallback_title)
        return title, text

    def try_get(u: str, t: int) -> Tuple[str, str]:
        r = requests.get(u, headers=headers, timeout=t, allow_redirects=True)
        r.raise_for_status()
        return parse_html(r.text, u)

    primary_errors: List[Exception] = []

    # Build candidate URL variants
    u = url.strip()
    candidates = []
    candidates.append(u)
    if u.endswith("/"):
        candidates.append(u[:-1])
    else:
        candidates.append(u + "/")
    if "://" in u:
        proto, rest = u.split("://", 1)
        if rest.startswith("www."):
            candidates.append(f"{proto}://{rest[len('www.'): ]}")
        else:
            candidates.append(f"{proto}://www.{rest}")

    # dedupe preserving order
    seen = set()
    variants = [v for v in candidates if not (v in seen or seen.add(v))]

    # Try primary GET on variants with small retries
    for variant in variants:
        for attempt in range(1, 3):  # 2 attempts per variant
            try:
                logger.info("Trying primary fetch: %s (attempt %d)", variant, attempt)
                title, text = try_get(variant, timeout)
                if text and len(text) >= 200:
                    logger.info("Primary fetch succeeded: %s (len=%d)", variant, len(text))
                    return title, text
                else:
                    primary_errors.append(ValueError(f"too-short text from {variant} len={len(text) if text else 0}"))
                    logger.info("Primary fetch returned too-short text for %s (len=%d)", variant, len(text) if text else 0)
            except Exception as e:
                logger.warning("Primary fetch error for %s: %s", variant, repr(e))
                primary_errors.append(e)
            time.sleep(0.4)

    # Fallback: use r.jina.ai text extraction proxy with retries (longer timeout)
    proxy_errors: List[Exception] = []
    proxy_prefix = "https://r.jina.ai/http://"
    proxy_target = url.replace("https://", "").replace("http://", "")
    proxy_url = proxy_prefix + proxy_target
    for attempt in range(1, 4):  # 3 attempts
        try:
            logger.info("Trying proxy fetch: %s (attempt %d)", proxy_url, attempt)
            rp = requests.get(proxy_url, headers=headers, timeout=30)
            rp.raise_for_status()
            proxy_text = rp.text.strip()
            if proxy_text and len(proxy_text) >= 120:
                title = url.rstrip("/").split("/")[-1] or url
                logger.info("Proxy fetch succeeded (len=%d)", len(proxy_text))
                return title, proxy_text
            else:
                proxy_errors.append(ValueError(f"proxy returned too-short text len={len(proxy_text) if proxy_text else 0}"))
                logger.info("Proxy returned too-short text (len=%d)", len(proxy_text) if proxy_text else 0)
        except Exception as e:
            logger.warning("Proxy fetch error: %s", repr(e))
            proxy_errors.append(e)
        time.sleep(1 * attempt)

    # All attempts failed
    raise RuntimeError(
        f"Failed to fetch/extract content from {url}\n"
        f"Primary examples: {[repr(e) for e in primary_errors[:3]]}\n"
        f"Proxy examples: {[repr(e) for e in proxy_errors[:3]]}"
    )


# -------------------------
# Chunking
# -------------------------
def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Naive character-based chunking."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


# -------------------------
# Embedding & FAISS helpers
# -------------------------
def get_embedder():
    """Load sentence-transformers embedder (singleton)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as ex:
            logger.error("sentence-transformers not installed or import failed: %s", ex)
            raise RuntimeError("sentence-transformers not installed in the container") from ex
        logger.info("Loading embedder model: %s", EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def ensure_index(embedding_dim: int = 384):
    """
    Load or create a FAISS IndexFlatIP index. Normalized vectors (unit length)
    so inner product equals cosine similarity.
    """
    global _index, _dim
    if _index is not None:
        return _index
    _dim = embedding_dim
    if os.path.exists(INDEX_PATH):
        try:
            _index = faiss.read_index(INDEX_PATH)
            logger.info("Loaded existing FAISS index from %s (ntotal=%d)", INDEX_PATH, _index.ntotal)
        except Exception as e:
            logger.warning("Failed to read existing FAISS index: %s. Creating a new one.", e)
            _index = faiss.IndexFlatIP(embedding_dim)
    else:
        _index = faiss.IndexFlatIP(embedding_dim)
        logger.info("Created new FAISS index with dim %d", embedding_dim)
    return _index


def persist_index_and_meta(meta: List[dict]):
    """Persist FAISS index and metadata to disk."""
    global _index
    if _index is None:
        logger.warning("persist_index_and_meta called but index is None")
        return
    try:
        faiss.write_index(_index, INDEX_PATH)
        pickle.dump(meta, open(META_PATH, "wb"))
        logger.info("Persisted index (%s) and meta (%s)", INDEX_PATH, META_PATH)
    except Exception as e:
        logger.error("Failed to persist index/meta: %s", e)


def load_meta() -> List[dict]:
    """Load persisted metadata list or return empty list."""
    if os.path.exists(META_PATH):
        try:
            meta = pickle.load(open(META_PATH, "rb"))
            return meta
        except Exception as e:
            logger.warning("Failed to load meta.pkl: %s", e)
            return []
    return []


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed list[str] -> np.ndarray shape (n, dim) dtype float32, normalized.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    embedder = get_embedder()
    vecs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs = vecs / norms
    return vecs.astype("float32")


def add_to_index(vectors: np.ndarray, metadatas: List[dict]):
    """
    Add vectors and corresponding metadatas to FAISS index and persist.
    """
    if vectors is None or vectors.size == 0:
        logger.info("No vectors to add_to_index.")
        return
    embedding_dim = vectors.shape[1]
    idx = ensure_index(embedding_dim)
    idx.add(vectors)
    meta = load_meta()
    meta.extend(metadatas)
    persist_index_and_meta(meta)


def query_index(query_vec: np.ndarray, top_k: int = 5) -> List[dict]:
    """
    Query FAISS index with normalized query vector (float32). Returns top_k metadata dicts.
    """
    idx = ensure_index()
    if idx.ntotal == 0:
        return []
    D, I = idx.search(np.array([query_vec]).astype("float32"), top_k)
    meta = load_meta()
    results = []
    for i_pos in I[0]:
        if i_pos < len(meta):
            results.append(meta[i_pos])
    return results


# -------------------------
# High-level ingestion helper
# -------------------------
def ingest_url_to_index(url: str, max_chunk_chars: int = 1000, sleep_between_batches: float = 0.08) -> dict:
    """
    Fetch the URL, chunk, embed (batched), add to FAISS index, persist.
    Returns dict: {"status":..., "url":..., "chunks":N}
    """
    logger.info("ingest_url_to_index: %s", url)
    title, text = fetch_text_from_url(url)
    if not text:
        return {"status": "empty", "url": url}
    chunks = chunk_text(text, max_chars=max_chunk_chars)
    if not chunks:
        return {"status": "no_chunks", "url": url}

    # embed in batches to avoid spikes
    vectors_list = []
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vecs = embed_texts(batch)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            raise
        vectors_list.append(vecs)
        time.sleep(sleep_between_batches)

    if vectors_list:
        vectors = np.vstack(vectors_list)
    else:
        vectors = np.zeros((0, 0), dtype="float32")

    metadatas = [{"title": title, "url": url, "snippet": chunk[:1200]} for chunk in chunks]
    add_to_index(vectors, metadatas)
    logger.info("Ingested %d chunks for %s", len(chunks), url)
    return {"status": "ingested", "url": url, "chunks": len(chunks)}


# -------------------------
# Query helper
# -------------------------
def query_by_text(query: str, top_k: int = 5) -> List[dict]:
    """Embed query with same embedder and query FAISS index."""
    embedder = get_embedder()
    qvec = embedder.encode([query], convert_to_numpy=True)[0]
    qvec = qvec / (np.linalg.norm(qvec) + 1e-10)
    qvec = qvec.astype("float32")
    return query_index(qvec, top_k=top_k)


# quick debug helper when executed inside the container
if __name__ == "__main__":
    logger.info("ingestion_utils debug mode")
    logger.info("DATA_DIR=%s", DATA_DIR)
    logger.info("INDEX_PATH exists=%s", os.path.exists(INDEX_PATH))
    logger.info("META_PATH exists=%s", os.path.exists(META_PATH))
