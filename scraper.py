#!/usr/bin/env python3
"""
Elite Dangerous Wiki Scraper (Threaded, Fast, API-based)
========================================================

- Fetches ALL pages
- Uses MediaWiki API (no 403)
- Multithreaded for speed
- Retry + error handling
- Safe for large datasets

Output:
    elite_wiki_data.json
"""

import json
import time
import re
import os
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import requests

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_URL = "https://elite-dangerous.fandom.com"
API_URL = f"{BASE_URL}/api.php"

REQUEST_DELAY = 0.1
MAX_WORKERS = 10  # ⚠️ 5–15 ist sinnvoll, mehr = riskant

OUTPUT_FILE = "elite_wiki_data.json"
DATA_DIR = "data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) " "Gecko/20100101 Firefox/149.0"
    )
}


logging.basicConfig(
    filename="scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────
# Session (thread-safe via per-thread usage)
# ─────────────────────────────────────────────
thread_local = threading.local()


def get_session() -> requests.Session:
    """Create one session per thread."""
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        thread_local.session = session
    return thread_local.session


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def get_all_page_titles() -> List[str]:
    """Fetch ALL page titles."""
    titles: List[str] = []

    session = get_session()

    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": 500,
        "apnamespace": 0,
        "format": "json",
    }

    logging.info("Fetching ALL page titles...")

    while True:
        resp = session.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        batch = data["query"]["allpages"]
        titles.extend(p["title"] for p in batch)

        logging.info(f"→ {len(titles)} titles collected")

        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
            time.sleep(REQUEST_DELAY)
        else:
            break

    logging.info(f"✓ Total pages: {len(titles)}")
    return titles


def clean_text(text: str) -> str:
    """Remove HTML tags."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_filename(title: str) -> str:
    """Create a filesystem-safe, unique filename for a given page title.

    Replaces spaces with underscores, removes unsafe chars, trims length,
    and appends a counter if a file with the same name already exists.
    """
    name = title.strip().replace(" ", "_")
    # allow alphanumerics, underscore, dash and dot
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    if not name:
        name = "page"
    maxlen = 200
    if len(name) > maxlen:
        name = name[:maxlen]

    filename = f"{name}.json"
    base = filename[:-5]
    path = os.path.join(DATA_DIR, filename.lower())
    counter = 1
    while os.path.exists(path):
        filename = f"{base}-{counter}.json"
        path = os.path.join(DATA_DIR, filename)
        counter += 1

    return filename


def fetch_page(title: str) -> Optional[Dict]:
    """Fetch a single page with retry logic."""
    session = get_session()

    params = {
        "action": "parse",
        "page": title,
        "prop": "text|categories",
        "format": "json",
    }

    for attempt in range(3):
        try:
            resp = session.get(API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                return None

            html = data["parse"]["text"]["*"]
            text = clean_text(html)

            categories = [
                c["*"].replace("Category:", "")
                for c in data["parse"].get("categories", [])
            ]

            return {
                "title": title,
                "url": f"{BASE_URL}/wiki/{title.replace(' ', '_')}",
                "content": text,
                "categories": categories,
            }

        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))

    logging.error(f"⚠ Failed after retries: {title}")
    return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main() -> None:
    titles = get_all_page_titles()

    total = len(titles)

    os.makedirs(DATA_DIR, exist_ok=True)

    logging.info(f"\nStarting threaded scraping with {MAX_WORKERS} workers...\n")

    saved = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_page, title): title for title in titles}

        for i, future in enumerate(as_completed(futures), 1):
            title = futures[future]

            try:
                result = future.result()
                if result and result["content"]:
                    try:
                        filename = safe_filename(result["title"])
                        path = os.path.join(DATA_DIR, filename)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        saved += 1
                    except Exception as exc:
                        logging.error(f"⚠ Failed saving {title}: {exc}")
            except Exception as exc:
                logging.error(f"⚠ Error in {title}: {exc}")

            if i % 50 == 0:
                logging.info(f"[{i}/{total}] processed")

    logging.info(f"\nSaved {saved} pages into '{DATA_DIR}'")


if __name__ == "__main__":
    main()
