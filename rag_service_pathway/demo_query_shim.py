# demo_query_shim.py -- simple keyword matching over meta.pkl
import os, pickle, re
DATA_DIR="/app/data"
META_PATH=os.path.join(DATA_DIR, "meta.pkl")

def demo_query_simple(q, k=5):
    if not os.path.exists(META_PATH):
        return []
    meta = pickle.load(open(META_PATH, "rb"))
    ql = q.lower()
    tokens = re.findall(r'\w+', ql)
    scored = []
    for item in meta:
        text = (item.get("title","") + " " + item.get("snippet","") + " " + item.get("url","")).lower()
        score = 0
        for tok in tokens:
            score += text.count(tok)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        results = [s[1] for s in scored if s[0] > 0][:k]
    else:
        results = [s[1] for s in scored][:k]
    return results
