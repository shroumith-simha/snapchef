# realtime_retriever.py
"""
Live retriever for Hebbar's Kitchen (or other site) that:
- downloads sitemap (cached)
- selects candidate URLs for a query (keyword heuristics)
- fetches pages (with polite headers), extracts text (readability)
- embeds page texts (sentence-transformers) and query, ranks by cosine
- returns top-N recipe metadata (title, url, snippet)

This variant excludes media/upload URLs (images) and uses a stricter recipe filter
so we return page links (recipes) rather than images which cause Cloudflare hotlink errors.
"""

import os
import time
import pickle
import requests
from readability import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from urllib.parse import urlparse
from typing import List, Dict

# CONFIG
DATA_DIR = "/app/data/realtime"
SITEMAP_CACHE = os.path.join(DATA_DIR, "sitemap_urls.pkl")
PAGE_CACHE_DIR = os.path.join(DATA_DIR, "pages")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CANDIDATE_FETCH = int(os.environ.get("CANDIDATE_FETCH", 10))
MAX_CHUNK_CHARS = 1200
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".bmp")

os.makedirs(PAGE_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# singleton embedder
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def url_looks_like_media(u: str) -> bool:
    u = u.split("?")[0].lower()
    if "/wp-content/uploads/" in u:
        return True
    if any(u.endswith(ext) for ext in IMAGE_EXTS):
        return True
    return False

def url_to_filename(url: str) -> str:
    fn = url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_")
    fn = "".join(c for c in fn if c.isalnum() or c in "_-.")
    return fn[:200]

def fetch_text(url: str, timeout: int = 15) -> Dict:
    """
    Fetch URL and extract title + main text using readability + BeautifulSoup fallback.
    Skip media URLs immediately. Cache extracted result to disk (PAGE_CACHE_DIR).
    """
    if url_looks_like_media(url):
        return {"url": url, "title": "", "text": ""}

    key = url_to_filename(url)
    cache_path = os.path.join(PAGE_CACHE_DIR, key + ".pkl")
    # return cache if present
    if os.path.exists(cache_path):
        try:
            return pickle.load(open(cache_path, "rb"))
        except Exception:
            pass

    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        doc = Document(html)
        title = (doc.short_title() or "").strip()
        content = doc.summary() or ""
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n").strip()
        if not text:
            soup2 = BeautifulSoup(html, "html.parser")
            text = soup2.get_text(separator="\n").strip()
        out = {"url": url, "title": title, "text": text}
        try:
            pickle.dump(out, open(cache_path, "wb"))
        except Exception:
            pass
        time.sleep(0.15)
        return out
    except Exception:
        # on any fetch/parsing issue return empty text (caller handles it)
        return {"url": url, "title": "", "text": ""}

def is_recipe_url(u: str) -> bool:
    """
    Stricter recipe heuristic:
    - must contain 'recipe' or '/recipes/' or 'recipes-' OR have a path length that looks like a post
    - exclude media/upload paths
    """
    u_low = u.lower()
    if url_looks_like_media(u_low):
        return False
    if "recipe" in u_low or "/recipes/" in u_low or "recipes-" in u_low or "recipe-" in u_low:
        return True
    # fallback heuristic: pages with multiple path segments and ending not in a file ext
    p = urlparse(u_low).path
    segments = [s for s in p.split("/") if s]
    if len(segments) >= 2 and not any(p.endswith(ext) for ext in IMAGE_EXTS):
        # could be a post URL - consider as candidate but lower priority
        return True
    return False

def load_sitemap_urls(sitemap_index_url: str) -> List[str]:
    """
    Download sitemap index, collect <loc> entries from child sitemaps, filter to the same host,
    and exclude media URLs. Cache result to reduce network calls.
    """
    if os.path.exists(SITEMAP_CACHE):
        try:
            data = pickle.load(open(SITEMAP_CACHE, "rb"))
            if time.time() - data.get("fetched_at", 0) < 24 * 3600:
                return data.get("urls", [])
        except Exception:
            pass

    sitemap_list = []
    try:
        r = requests.get(sitemap_index_url, headers=REQUEST_HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        sitemap_list = [loc.text.strip() for loc in soup.find_all("loc") if loc.text and loc.text.strip()]
    except Exception:
        sitemap_list = []

    all_locs = []
    for s in sitemap_list:
        try:
            r = requests.get(s, headers=REQUEST_HEADERS, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
            locs = [loc.text.strip() for loc in soup.find_all("loc") if loc.text and loc.text.strip()]
            all_locs.extend(locs)
            time.sleep(0.05)
        except Exception:
            continue

    base_host = urlparse(sitemap_index_url).netloc
    filtered = []
    for u in all_locs:
        try:
            pu = urlparse(u)
            if pu.netloc == base_host and is_recipe_url(u):
                # final filter: exclude media
                if not url_looks_like_media(u):
                    filtered.append(u)
        except Exception:
            continue
    filtered = sorted(set(filtered))
    try:
        pickle.dump({"base": sitemap_index_url, "urls": filtered, "fetched_at": time.time()}, open(SITEMAP_CACHE, "wb"))
    except Exception:
        pass
    return filtered

def select_candidates(query: str, sitemap_urls: List[str], top_n: int = CANDIDATE_FETCH) -> List[str]:
    """
    Heuristic to select candidate URLs:
    - prefer URLs containing query tokens and 'recipe' in path
    - deprioritize generic or media-like URLs (we already filtered media)
    """
    q_tokens = [t for t in query.lower().split() if t]
    scored = []
    for u in sitemap_urls:
        score = 0
        ul = u.lower()
        # boost if 'recipe' in url
        if "recipe" in ul:
            score += 3
        for t in q_tokens:
            score += ul.count(t) * 2
        last = ul.rstrip("/").split("/")[-1]
        for t in q_tokens:
            if t in last:
                score += 4
        if score > 0:
            scored.append((score, u))
    if scored:
        scored.sort(reverse=True)
        return [u for _, u in scored[:top_n]]
    # fallback: return first top_n sitemap urls
    return sitemap_urls[:top_n]

def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    if not text:
        return []
    text = text.strip()
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        # try to cut at newline boundary if possible
        chunk = text[i:end]
        chunks.append(chunk.strip())
        i = end
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    emb = get_embedder()
    vecs = emb.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs = vecs / norms
    return vecs.astype("float32")

def query_live(query: str, sitemap_index_url: str = "https://hebbarskitchen.com/sitemap.xml", k: int = 5) -> List[Dict]:
    """
    Main entry: returns top-k results: {title, url, snippet}
    """
    sitemap_urls = load_sitemap_urls(sitemap_index_url)
    if not sitemap_urls:
        return []

    candidates = select_candidates(query, sitemap_urls, top_n=CANDIDATE_FETCH)
    # debug log
    print(f"[query_live] Query='{query}' candidates: {candidates}")

    docs = []
    for u in candidates:
        print(f"[query_live] Fetching: {u}")
        page = fetch_text(u)
        title = page.get("title") or ""
        text = page.get("text") or ""
        snippet = (text[:1000] + "...") if len(text) > 1000 else text
        # only keep docs that actually contain text
        if text.strip():
            docs.append({"url": u, "title": title, "text": text, "snippet": snippet})
        else:
            print(f"[query_live] Skipped (no text): {u}")

    if not docs:
        return []

    # embed query and doc snippets
    emb = get_embedder()
    query_emb = emb.encode([query], convert_to_numpy=True)[0]
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

    texts = [d["snippet"] or d["title"] or "" for d in docs]
    doc_vecs = embed_texts(texts)
    sims = (doc_vecs @ query_emb).astype("float32")
    ranked_idx = np.argsort(-sims)[:k]

    results = []
    for idx in ranked_idx:
        if idx < len(docs):
            d = docs[int(idx)]
            results.append({"title": d["title"] or d["url"], "url": d["url"], "snippet": d["snippet"]})
    return results
