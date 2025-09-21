# app.py
"""
Flask wrapper for the Pathway + FAISS RAG microservice (demo + live retriever).

Endpoints:
- GET  /health
- POST /ingest_url    -> {"url": "<page_url>"}  (ingest single page)
- POST /query_recipe  -> {"query": "<text>", "k": 5}  (uses demo shim if present)
- POST /ingest_seeds  -> optional body: {"delay_between": 2.0, "max_retries": 2} (starts background ingestion)
- POST /query_live    -> {"query":"...","k":5}  (live-scrape + embed from Hebbar's sitemap)
"""

import os
import threading
from flask import Flask, request, jsonify

# ingestion/query helpers (real implementation)
from ingestion_utils import ingest_url_to_index, query_by_text

# seed ingestion helper (optional)
try:
    from seed_ingest import ingest_seed_list
except Exception:
    ingest_seed_list = None  # endpoint will return error if called and seed_ingest isn't present

# realtime retriever (live scraping + per-query embedding)
try:
    from realtime_retriever import query_live
except Exception:
    query_live = None  # route will return error if missing


# -------------------------
# Pathway safe start (non-blocking)
# -------------------------
def start_pathway_bg():
    """
    Try to start the Pathway computation in a background thread if available.
    This is tolerant so missing/changed pathway SDK versions won't crash the web process.
    """
    def _run():
        try:
            import pathway as pw
        except Exception:
            print("pathway not importable — skipping pipeline startup")
            return
        try:
            if hasattr(pw, "run"):
                print("Pathway available — calling pw.run()")
                pw.run()
            else:
                print("pathway installed but no pw.run() — skipping pipeline startup")
        except Exception as e:
            print("Exception while running Pathway pipeline:", e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("Pathway pipeline started in background thread.")


# Start Pathway (non-blocking). Comment out if you don't want it to start automatically.
start_pathway_bg()

app = Flask(__name__)


# -------------------------
# Routes
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return "ok"


@app.route("/ingest_url", methods=["POST"])
def api_ingest_url():
    """
    Ingest a single URL immediately.
    Body: JSON {"url": "<page_url>"}
    """
    data = request.json or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "missing 'url'"}), 400
    try:
        res = ingest_url_to_index(url)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query_recipe", methods=["POST"])
def api_query_recipe():
    """
    Query the index (demo). Body: {"query": "<text>", "k": <int>}
    Tries demo_query_shim first (fast). Falls back to the real FAISS query_by_text.
    """
    data = request.json or {}
    q = data.get("query") or data.get("q")
    k = int(data.get("k", 5))
    if not q:
        return jsonify({"error": "missing 'query'"}), 400
    try:
        # Try demo shim first (very fast, deterministic)
        try:
            from demo_query_shim import demo_query_simple
            results = demo_query_simple(q, k=k)
            return jsonify({"query": q, "results": results})
        except Exception:
            # If demo shim not available or fails, use real vector query
            results = query_by_text(q, top_k=k)
            return jsonify({"query": q, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ingest_seeds", methods=["POST"])
def api_ingest_seeds():
    """
    Trigger batch ingestion of seeds_indian.txt.
    Optional JSON body: {"delay_between": 2.0, "max_retries": 2}
    This implementation runs ingestion in a background thread and returns immediately.
    """
    if ingest_seed_list is None:
        return jsonify({"error": "seed_ingest not available on this service"}), 500

    data = request.json or {}
    delay_between = float(data.get("delay_between", 2.0))
    max_retries = int(data.get("max_retries", 2))

    def _bg():
        try:
            print("Background seed ingestion started")
            ingest_seed_list(delay_between=delay_between, max_retries=max_retries)
            print("Background seed ingestion finished")
        except Exception as e:
            print("seed ingest error:", e)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "seed ingestion running in background"})


@app.route("/query_live", methods=["POST"])
def api_query_live():
    """
    Live web-scrape + RAG over Hebbar's Kitchen sitemap (or other sitemap).
    Body: {"query":"paneer butter masala", "k":5}
    """
    if query_live is None:
        return jsonify({"error": "realtime_retriever not available on this service"}), 500

    data = request.json or {}
    q = data.get("query") or data.get("q")
    k = int(data.get("k", 5))
    if not q:
        return jsonify({"error": "missing 'query'"}), 400
    try:
        # You can optionally pass a different sitemap_index_url in data (e.g. "sitemap": "...")
        sitemap_url = data.get("sitemap") or "https://hebbarskitchen.com/sitemap.xml"
        results = query_live(q, sitemap_index_url=sitemap_url, k=k)
        return jsonify({"query": q, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Local debug run
# -------------------------
if __name__ == "__main__":
    # Only used when running app.py directly (not under gunicorn in Docker)
    app.run(host="0.0.0.0", port=8000)
