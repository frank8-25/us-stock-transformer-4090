"""Document preparation and optional cached body retrieval; never overwrite raw CSVs."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse

import pandas as pd

SOURCES = ("news", "earnings_call", "ten_k")
RAW_NAMES = {"news": "news", "earnings_call": "earnings_call", "ten_k": "10k"}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


class BodyParser(HTMLParser):
    """Prefer article/main content; discard scripts, navigation and hidden XBRL."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.article = []
        self.focus = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        blocked = tag in {"script", "style", "nav", "header", "footer", "noscript", "svg",
                          "ix:header", "ix:hidden", "head"}
        blocked |= "hidden" in attrs or attrs.get("aria-hidden") == "true"
        blocked |= "display:none" in attrs.get("style", "").replace(" ", "").lower()
        blocked |= bool(self.stack and self.stack[-1][1])
        focus = tag in {"article", "main"}
        if tag not in {"br", "hr", "img", "input", "meta", "link", "source", "wbr", "area", "embed"}:
            self.stack.append((tag, blocked, focus))
            self.focus += int(focus)
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"} and not blocked:
            self.parts.append("\n")
            if self.focus:
                self.article.append("\n")

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.focus -= sum(int(item[2]) for item in self.stack[index:])
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(data + " ")
            if self.focus:
                self.article.append(data + " ")

    def text(self):
        content = self.article if len("".join(self.article).strip()) >= 200 else self.parts
        return "\n".join(" ".join(line.split()) for line in "".join(content).splitlines() if line.strip())


def fetch_body(url, cache_dir, user_agent, delay=1.0, retry_errors=False):
    """Fetch public HTML/PDF only. Failed retrievals are recorded, never title fallbacks."""
    cache = Path(cache_dir) / (digest(url) + ".json")
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if not saved["error"] or not retry_errors:
            return saved
    result = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(),
              "text": "", "error": "", "extractor": "stdlib-html-v1"}
    try:
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("Only HTTP(S) document URLs are supported.")
        if not user_agent.strip():
            raise ValueError("Set --user-agent with a contact identity before fetching.")
        import requests
        time.sleep(max(0, delay))
        with requests.get(url, headers={"User-Agent": user_agent}, timeout=(10, 45), stream=True) as response:
            response.raise_for_status()
            payload = bytearray()
            for block in response.iter_content(65536):
                payload.extend(block)
                if len(payload) > 30 * 1024 * 1024:
                    raise ValueError("Document exceeds 30 MB download limit.")
            result["resolved_url"] = response.url
            if bytes(payload).startswith(b"%PDF"):
                from pypdf import PdfReader
                result["text"] = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(payload)).pages)
                result["extractor"] = "pypdf"
            else:
                parser = BodyParser()
                response._content = bytes(payload)
                response.encoding = response.apparent_encoding or "utf-8"
                parser.feed(response.text)
                result["text"] = parser.text()
        if len(result["text"].strip()) < 200:
            raise ValueError("Extracted body is too short; inspect access restrictions or extraction.")
        if re.search(r"(enable javascript|verify you are human|access denied)", result["text"][:500], re.I):
            raise ValueError("Access challenge detected; body requires manual review.")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["text"] = ""
    save_json(cache, result)
    return result


DOC_COLUMNS = ["document_id", "source_type", "date", "available_date", "date_status",
               "title", "url", "text", "text_origin", "quality_flags", "preparation_error"]


def prepare_documents(root, ticker, output, start, end, fetch=False, user_agent="",
                      allow_title_only=False, allow_incomplete=False, allow_unverified=False,
                      overrides=None, limit=None, delay=1.0, retry_errors=False):
    output = Path(output)
    if fetch and not user_agent.strip():
        raise ValueError("--fetch-body requires --user-agent with a contact identity.")
    overrides_df = read_csv(overrides).astype(str) if overrides else pd.DataFrame()
    override_map = {}
    if not overrides_df.empty:
        required = {"document_id", "available_date", "evidence"}
        if not required.issubset(overrides_df.columns):
            raise ValueError(f"Date override CSV requires {sorted(required)}")
        if overrides_df.document_id.duplicated().any():
            raise ValueError("Duplicate document IDs in date overrides.")
        override_map = overrides_df.set_index("document_id").to_dict("index")
    records = []
    source_inventory = {}
    for source in SOURCES:
        path = Path(root) / "data/raw" / source / f"{ticker}_{RAW_NAMES[source]}_raw.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = read_csv(path).astype(str)
        required = {"date", "title", "url"} | ({"content"} if source != "news" else set())
        if not required.issubset(raw.columns):
            raise ValueError(f"Missing raw columns in {path}: {required - set(raw.columns)}")
        parsed = pd.to_datetime(raw.date, errors="raise")
        raw = raw.loc[(parsed >= pd.Timestamp(start)) & (parsed <= pd.Timestamp(end))].drop_duplicates()
        raw = raw.sort_values(["date", "url", "title"], kind="stable")
        if limit is not None:
            raw = raw.head(limit)
        source_inventory[source] = len(raw)
        for row in raw.to_dict("records"):
            identity = json.dumps([ticker, source, row["date"], row["url"], row["title"],
                                   row.get("content", "")], ensure_ascii=False)
            document_id = digest(identity)[:24]
            body = row.get("content", "").strip()
            text_origin = "supplied_body"
            flags, errors = [], []
            date_status = "source_date"
            available_date = row["date"]
            if source == "earnings_call" and "CFO Commentary" in row["title"]:
                date_status = "unverified"
                flags.append("cfo_commentary_publication_date_unverified")
            if document_id in override_map:
                override = override_map[document_id]
                if not override["evidence"].strip():
                    raise ValueError("Date override requires nonempty evidence.")
                available_date = pd.Timestamp(override["available_date"]).strftime("%Y-%m-%d")
                date_status = "reviewed_override"
            if date_status == "unverified" and not allow_unverified:
                errors.append("Publication date needs review; supply --date-overrides.")
            if fetch:
                fetched = fetch_body(row["url"], Path(root) / "data/cache/article_bodies",
                                     user_agent, delay, retry_errors)
                if fetched["error"]:
                    errors.append(fetched["error"])
                else:
                    body = fetched["text"]
                    text_origin = "fetched_body"
                    flags.append("automated_extraction_requires_quality_review")
            if not body:
                if source == "news" and allow_title_only and not fetch:
                    body = row["title"]
                    text_origin = "title_only"
                    flags.append("title_only")
                else:
                    errors.append("No body: use --fetch-body or provide a complete content column.")
            elif text_origin == "supplied_body" and source != "news":
                if len(body) == 20000 or "us-gaap:" in body or "xbrli:" in body:
                    flags.append("possibly_truncated_or_xbrl_contaminated")
                    if not allow_incomplete:
                        errors.append("Supplied body needs replacement/cleaning; use --fetch-body.")
            records.append(dict(document_id=document_id, source_type=source, date=row["date"],
                                available_date=available_date, date_status=date_status,
                                title=row["title"], url=row["url"], text=body, text_origin=text_origin,
                                quality_flags=";".join(flags), preparation_error="; ".join(errors)))
    documents = pd.DataFrame(records, columns=DOC_COLUMNS)
    write_csv(documents, output / "documents.csv")
    write_csv(documents.loc[documents.preparation_error.ne("")], output / "preparation_errors.csv")
    save_json(output / "preparation_summary.json",
              {"rows": len(documents), "ready": int(documents.preparation_error.eq("").sum()),
               "failed": int(documents.preparation_error.ne("").sum()), "sources": source_inventory,
               "scope": "sample" if limit is not None else "full"})
    return documents


def chunk_text(text, tokenizer, max_tokens=512, max_chars=1800):
    """Fit the exact existing prompt without dropping text at either model limit."""
    from fingpt_sentiment import build_fingpt_prompt
    remaining = " ".join(text.split())
    chunks = []
    while remaining:
        low, high, best = 1, min(len(remaining), max_chars), 0
        while low <= high:
            mid = (low + high) // 2
            length = len(tokenizer.encode(build_fingpt_prompt(remaining[:mid]), add_special_tokens=True))
            if length <= max_tokens:
                best, low = mid, mid + 1
            else:
                high = mid - 1
        if not best:
            raise ValueError("Prompt exceeds token budget.")
        boundary = remaining.rfind(" ", 0, best + 1)
        if best < len(remaining) and boundary > best // 2:
            best = boundary
        chunks.append(remaining[:best].strip())
        remaining = remaining[best:].lstrip()
    if not chunks:
        raise ValueError("Empty document.")
    return chunks
