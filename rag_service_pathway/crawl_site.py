# crawl_site.py
"""
Simple site crawler to discover recipe-like pages under a domain.

Usage (from project root):
  python rag_service_pathway\crawl_site.py https://hebbarskitchen.com  --out rag_service_pathway\seeds_indian.txt --max-pages 200

The crawler:
- starts from the base URL,
- follows internal links (same host),
- filters candidate URLs using simple heuristics (contains 'recipe', '/recipe-', endswith '.html', etc.),
- deduplicates URLs and writes them to the output file (one URL per line).
"""
import sys
import time
import re
import argparse
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from collections import deque

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def is_same_domain(base_netloc, url):
    try:
        p = urlparse(url)
        return p.netloc == base_netloc or p.netloc.endswith("." + base_netloc)
    except:
        return False

def normalize(u):
    return u.split("#")[0].rstrip("/")

def find_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        links.append(urljoin(base_url, href))
    return links

def looks_like_recipe(url):
    u = url.lower()
    # heuristics: contains 'recipe', '/recipes/', endswith .html, '/recipes-' etc.
    patterns = [
        r"/recipe",
        r"/recipes/",
        r"recipe-",
        r"/recipes-",
        r"/recipies",  # sometimes misspellings
        r"/food/recipes",
        r"\.html$"
    ]
    for p in patterns:
        if re.search(p, u):
            return True
    return False

def crawl(start_url, max_pages=200, delay=0.8):
    parsed = urlparse(start_url)
    base_netloc = parsed.netloc
    seen = set()
    found_recipes = set()
    q = deque([start_url])
    seen.add(normalize(start_url))

    while q and len(seen) < max_pages:
        url = q.popleft()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                # skip non-200
                time.sleep(delay)
                continue
            html = resp.text
            # extract candidate links
            for link in find_links(html, url):
                linkn = normalize(link)
                if not linkn:
                    continue
                if linkn in seen:
                    continue
                if not is_same_domain(base_netloc, linkn):
                    continue
                seen.add(linkn)
                # if recipe-like, record
                if looks_like_recipe(linkn):
                    found_recipes.add(linkn)
                # also continue crawling for more pages
                q.append(linkn)
            time.sleep(delay)
        except Exception as e:
            # ignore and continue
            # print("crawl error:", e)
            time.sleep(delay)
            continue
    return list(found_recipes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_url", help="site root to crawl, e.g. https://hebbarskitchen.com")
    parser.add_argument("--out", default="rag_service_pathway/seeds_indian.txt", help="output file for seed URLs")
    parser.add_argument("--max-pages", type=int, default=300, help="max pages to visit")
    parser.add_argument("--delay", type=float, default=0.8, help="delay between requests")
    args = parser.parse_args()

    print("Crawling:", args.start_url)
    recipes = crawl(args.start_url, max_pages=args.max_pages, delay=args.delay)
    recipes = sorted(set(recipes))
    print("Found", len(recipes), "candidate recipe URLs. Writing to", args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for u in recipes:
            f.write(u + "\n")
    print("Done.")

if __name__ == "__main__":
    main()
