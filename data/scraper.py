import time
import pandas as pd
import cloudscraper  # handles Cloudflare anti-bot checks
from bs4 import BeautifulSoup

BASE_URL = "https://exrx.net"

# helper to fetch a page and return parsed BeautifulSoup

# create one cloudscraper session to reuse
_scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows"}
)

def fetch_soup(url, delay=1, timeout=10):
    try:
        resp = _scraper.get(url, timeout=timeout)
        resp.raise_for_status()
        time.sleep(delay)
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        # propagate so callers can decide what to do
        raise


# start by fetching the main directory page using requests
soup = fetch_soup(BASE_URL + "/Lists/Directory")

# 1️⃣ zbierz linki do stron mięśni (ExList pages)
import urllib.parse

# helper to normalize a raw href into a valid /Lists/ExList/... URL

def normalize_muscle_href(href: str) -> str:
    # drop fragments
    href = href.split("#")[0]
    # build absolute URL first
    full = urllib.parse.urljoin(BASE_URL, href)
    parsed = urllib.parse.urlparse(full)
    path = parsed.path
    # ensure path has /Lists/ExList prefix
    if "ExList/" in path and "/Lists/ExList" not in path:
        # insert "/Lists" before the first occurrence of ExList
        idx = path.find("ExList")
        path = "/Lists/" + path[idx:]
    # rebuild URL
    normalized = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return normalized

muscle_links = set()
for a in soup.find_all("a", href=True):
    raw = a["href"]
    if "ExList/" in raw:
        full = normalize_muscle_href(raw)
        muscle_links.add(full)

muscle_links = list(muscle_links)
print(f"Found {len(muscle_links)} muscle groups")

data = []

# 2️⃣ przejdź po każdej grupie mięśni
for idx, muscle_url in enumerate(muscle_links, start=1):
    print(f"[{idx}/{len(muscle_links)}] fetching muscle page {muscle_url}")
    muscle_soup = fetch_soup(muscle_url, delay=0.5, timeout=10)

    # extract name from URL path (last component)
    muscle_name = muscle_url.rstrip("/").split("/")[-1]

    # 3️⃣ znajdź ćwiczenia na stronie
    for a in muscle_soup.find_all("a", href=True):
        href = a["href"]
        if "/WeightExercises/" in href:
            exercise_url = urllib.parse.urljoin(BASE_URL, href)
            data.append({
                "muscle": muscle_name,
                "exercise_url": exercise_url
            })

print(f"Collected {len(data)} exercises")

# build DataFrame and deduplicate urls
raw_df = pd.DataFrame(data).drop_duplicates()

# helper function to parse muscle categories from an exercise page
import re

def parse_muscles_from_soup(soup):
    muscles = {}
    for h2 in soup.find_all("h2"):
        if "Muscles" in h2.text:
            node = h2.find_next_sibling()
            cat = None
            while node and node.name not in ["h1", "h2", "h3"]:
                if node.name == "p" and node.find("strong"):
                    # category label (e.g. "Target", "Synergists")
                    cat = node.text.strip()
                    # normalise category label by removing trailing colon
                    cat = re.sub(r":$", "", cat)
                    muscles[cat] = []
                elif node.name == "ul" and cat:
                    muscles[cat].extend([li.text.strip() for li in node.find_all("li")])
                node = node.find_next_sibling()
            break
    return muscles

# iterate unique exercises and fetch their muscle category info
exercise_records = []
seen = set()
for idx, (_, row) in enumerate(raw_df.iterrows(), start=1):
    url = row["exercise_url"]
    if url in seen:
        continue
    seen.add(url)
    print(f"[{idx}/{len(raw_df)}] fetching {url}")
    try:
        ex_soup = fetch_soup(url, delay=0.3, timeout=10)
    except Exception as exc:
        print(f"  -> failed to fetch {url}: {exc}")
        continue
    # get human-readable name if available
    title = ex_soup.find("h1")
    name = title.text.strip() if title else url.rstrip("/").split("/")[-1]

    muscles = parse_muscles_from_soup(ex_soup)
    # flatten muscle lists into semicolon-separated string
    record = {
        "exercise_name": name,
        "exercise_url": url,
    }
    for cat, vals in muscles.items():
        record[cat] = "; ".join(vals)
    exercise_records.append(record)

# build final DataFrame, ensuring consistent column order
ex_df = pd.DataFrame(exercise_records)

# optionally reorder columns according to desired category list
desired = [
    "exercise_name",
    "exercise_url",
    "Target",
    "Synergists",
    "Dynamic Stabilizers",
    "Stabilizers",
    "Antagonist Stabilizers",
]
# append any extra columns that were discovered
for col in ex_df.columns:
    if col not in desired:
        desired.append(col)

ex_df = ex_df.reindex(columns=[c for c in desired if c in ex_df.columns])

output_path = "data/exrx_exercises_muscles.csv"
ex_df.to_csv(output_path, index=False)
print(f"Wrote muscle worksheet to {output_path}, {len(ex_df)} exercises")

