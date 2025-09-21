# seed_ingest.py
import time
import os
from ingestion_utils import ingest_url_to_index

SEEDS_FILE = os.path.join(os.path.dirname(__file__), "seeds_indian.txt")

def ingest_seed_list(delay_between=2.0, max_retries=2):
    if not os.path.exists(SEEDS_FILE):
        print("No seeds file:", SEEDS_FILE)
        return {"error": "no_seeds"}
    results = []
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    for url in urls:
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[seed_ingest] ingesting {url} (attempt {attempt})")
                res = ingest_url_to_index(url)
                results.append({"url": url, "result": res})
                success = True
                break
            except Exception as e:
                print(f"[seed_ingest] error ingesting {url}: {e}")
                time.sleep(1 * attempt)
        if not success:
            results.append({"url": url, "result": "failed"})
        time.sleep(delay_between)
    return {"ingested": len([r for r in results if r["result"] and r["result"] != "failed"]), "details": results}

if __name__ == "__main__":
    print("Starting seed ingestion")
    out = ingest_seed_list()
    print("Done:", out)
