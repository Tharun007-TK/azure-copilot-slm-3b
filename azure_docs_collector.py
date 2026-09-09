"""
Azure docs collector.

Primary path: clone MicrosoftDocs/azure-docs (CC BY 4.0) and pull clean markdown.
Fallback path: scrape learn.microsoft.com for pages not covered by the repo
(rare — only needed for auto-generated reference pages like REST API specs
that aren't checked into the docs repo as markdown).

Why not scrape by default: the repo gives you the same content, pre-cleaned,
with YAML frontmatter (title, description, ms.service, ms.topic), versioned,
and legally unambiguous. HTML scraping learn.microsoft.com means fighting
their JS-rendered nav, rate limits, and a ToS you'd rather not test.
"""

import os
import re
import json
import time
import shutil
import subprocess
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

REPO_URL = "https://github.com/MicrosoftDocs/azure-docs.git"
CLONE_DIR = Path("./azure-docs-src")
OUTPUT_FILE = Path("./azure_docs_corpus.jsonl")

# Only pull articles, skip includes/media/toolchain noise
INCLUDE_DIRS_HINT = "articles"  # top-level dir in the repo holding actual docs


@dataclass
class DocRecord:
    source: str          # "github" or "web"
    path: str            # repo-relative path or URL
    service: str         # e.g. "azure-functions", "" if unknown
    title: str
    description: str
    content: str          # cleaned markdown/text body


def clone_or_update_repo(shallow: bool = True):
    if CLONE_DIR.exists():
        print(f"[collector] {CLONE_DIR} already exists, pulling latest")
        subprocess.run(["git", "-C", str(CLONE_DIR), "pull", "--ff-only"], check=True)
        return
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [REPO_URL, str(CLONE_DIR)]
    print(f"[collector] cloning {REPO_URL} (this repo is large, ~2-3GB shallow)")
    subprocess.run(cmd, check=True)


def parse_frontmatter(raw: str):
    """Split YAML frontmatter from markdown body. Azure docs use --- delimited blocks."""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    body = parts[2].strip()
    return meta, body


def clean_markdown(body: str) -> str:
    # Strip Azure docs' custom include directives, e.g. [!INCLUDE [x](../includes/x.md)]
    body = re.sub(r"\[!INCLUDE.*?\]\(.*?\)", "", body)
    # Strip zone pivots and note/tip/warning admonition markers, keep the text
    body = re.sub(r"^\s*>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]\s*", "", body, flags=re.MULTILINE)
    # Collapse 3+ blank lines
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def infer_service_from_path(path: Path) -> str:
    # e.g. articles/azure-functions/foo.md -> "azure-functions"
    parts = path.parts
    if INCLUDE_DIRS_HINT in parts:
        idx = parts.index(INCLUDE_DIRS_HINT)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def collect_from_repo(min_words: int = 150) -> list[DocRecord]:
    records = []
    articles_root = CLONE_DIR / "articles"
    if not articles_root.exists():
        raise FileNotFoundError(f"{articles_root} not found — did the clone succeed?")

    md_files = list(articles_root.rglob("*.md"))
    print(f"[collector] found {len(md_files)} markdown files")

    for i, f in enumerate(md_files):
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        meta, body = parse_frontmatter(raw)
        body = clean_markdown(body)

        if len(body.split()) < min_words:
            continue  # skip stubs/redirects

        rec = DocRecord(
            source="github",
            path=str(f.relative_to(CLONE_DIR)),
            service=infer_service_from_path(f),
            title=meta.get("title", ""),
            description=meta.get("description", ""),
            content=body,
        )
        records.append(rec)

        if i % 2000 == 0:
            print(f"[collector] processed {i}/{len(md_files)}")

    return records


def fallback_scrape(urls: list[str], delay_seconds: float = 1.0) -> list[DocRecord]:
    """
    Only use this for pages genuinely missing from the repo.
    Respects a fixed delay; does not bypass robots.txt or auth walls.
    """
    records = []
    headers = {"User-Agent": "azure-docs-collector-research/1.0"}

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[fallback] failed {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            continue

        for tag in main.select("nav, .feedback-section, .metadata, script, style"):
            tag.decompose()

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""
        content = main.get_text("\n", strip=True)

        if len(content.split()) < 150:
            continue

        records.append(DocRecord(
            source="web", path=url, service="", title=title,
            description="", content=content,
        ))
        time.sleep(delay_seconds)

    return records


def write_jsonl(records: list[DocRecord], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"[collector] wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    clone_or_update_repo(shallow=True)
    records = collect_from_repo()
    write_jsonl(records, OUTPUT_FILE)

    # Example fallback usage — only fill this with URLs you've confirmed
    # aren't already covered by the repo pull above.
    # fallback_records = fallback_scrape(["https://learn.microsoft.com/en-us/rest/api/..."])
    # write_jsonl(records + fallback_records, OUTPUT_FILE)
