"""Standalone LOCAL data pipeline for Galen-4B (download + clean + dedup + redact).

Run this on your laptop / workstation so the VM only has to do tokenization
(which needs the GPU). This script intentionally has NO `llm.*` imports — copy
it to any folder and it works as long as the two pip deps below are installed.

Output layout (category -> source -> stage):

    <out-dir>/
    +- medical/
    |  +- pubmed/
    |  |   +- raw/        shard_00000.jsonl.gz, shard_00001.jsonl.gz, ...
    |  |   +- cleaned/    cleaned_00000.jsonl.gz, ...   <- ship this to VM
    |  |   +- _DOWNLOAD_DONE
    |  |   +- _CLEAN_DONE
    |  +- pmc/
    |  +- ...
    +- general/
       +- wikipedia_en/
       |   +- raw/, cleaned/
       +- c4/
       +- ...

Each cleaned line is a JSON object: {"text": "...", "source": "...", "category": "..."}.
The cleaned/ subfolder is exactly the format `prepare_data.py` on the VM
expects -- once a source's `_CLEAN_DONE` marker exists the VM pipeline will
skip download / filter / dedup / redact and go straight to tokenization.

Sync to the VM after a run:

    rsync -avh --progress ./data/ user@vm:/path/to/project/data/local_prepared/

Then on the VM, point your training config at the cleaned dirs (or use the
helper config emitter at the bottom of this file -- pass --emit-config).

Install:
    pip install "datasets>=2.18" "huggingface_hub[hf_transfer]>=0.22"
    # Optional (better dedup / language id):
    pip install "datasketch>=1.6" "langdetect>=1.0.9"

Credentials:
    The script reads HF_TOKEN from (in order):
      1. --hf-token CLI flag
      2. environment variable HF_TOKEN
      3. a .env file in the script's folder OR the --credentials-file path

    .env format (one KEY=VALUE per line, no quotes needed):
      HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      # optional, for hf_transfer accelerated downloads
      HF_HUB_ENABLE_HF_TRANSFER=1

Usage:
    # Full pipeline, all sources
    python prepare_data_local.py --out-dir ./data

    # Just one source
    python prepare_data_local.py --out-dir ./data --source pubmed

    # Just medical sources
    python prepare_data_local.py --out-dir ./data --category medical

    # Smoke test (cap each source)
    python prepare_data_local.py --out-dir ./data --max-docs 1000

    # Download only (skip filter/dedup/redact -- do them later)
    python prepare_data_local.py --out-dir ./data --download-only

    # Filter/dedup/redact only (raw shards must already exist)
    python prepare_data_local.py --out-dir ./data --clean-only

    # List sources
    python prepare_data_local.py --list

    # Emit a YAML config snippet for the VM that points at the cleaned dirs
    python prepare_data_local.py --out-dir ./data --emit-config
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from ftplib import FTP
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)
log = logging.getLogger("prep")

SHARD_SIZE = 50_000

# ---------------------------------------------------------------------------
# Source list -- mirrors configs/data/medical_mix.yaml. Each entry carries:
#   name           folder name under <category>/
#   category       "medical" or "general"
#   kind           which downloader to use
#   args           kwargs for the downloader
#   weight         training-mix weight (informational, copied to _meta.json)
#   target_tokens  approximate token budget after dedup (informational)
# ---------------------------------------------------------------------------
SOURCES: list[dict[str, Any]] = [
    # ===== MEDICAL =====
    {
        "name": "pubmed", "category": "medical", "kind": "pubmed_ftp",
        "args": {"years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]},
        "weight": 0.1125, "target_tokens": 9_000_000_000,
    },
    # pmc/open_access is a script-based dataset, removed in `datasets>=3`.
    # No drop-in parquet replacement exists on the Hub. To re-enable, either pin
    # `datasets<3` (breaks streaming for c4/s2orc) or write a custom downloader
    # off PMC's FTP archive (similar to download_pubmed). Disabled until then.
    # {
    #     "name": "pmc", "category": "medical", "kind": "hf",
    #     "args": {"repo": "pmc/open_access", "split": "train", "streaming": True},
    #     "weight": 0.3125, "target_tokens": 25_000_000_000,
    # },
    {
        "name": "s2orc", "category": "medical", "kind": "hf",
        "args": {"repo": "leminda-ai/s2orc_small", "split": "train", "streaming": True},
        "weight": 0.125, "target_tokens": 10_000_000_000,
    },
    {
        "name": "guidelines", "category": "medical", "kind": "hf",
        "args": {"repo": "epfl-llm/guidelines", "split": "train"},
        "weight": 0.005, "target_tokens": 400_000_000,
    },
    {
        "name": "medbooks", "category": "medical", "kind": "hf",
        "args": {"repo": "FreedomIntelligence/medical-o1-reasoning-SFT",
                 "subset": "en", "split": "train"},
        "weight": 0.05, "target_tokens": 4_000_000_000,
    },
    {
        "name": "clinicaltrials", "category": "medical", "kind": "ctgov",
        "args": {"fields": ["BriefSummary", "DetailedDescription", "EligibilityCriteria"]},
        "weight": 0.00625, "target_tokens": 500_000_000,
    },
    {
        "name": "medqa", "category": "medical", "kind": "hf",
        "args": {"repo": "bigbio/med_qa", "subset": "med_qa_en_source",
                 "split": "train"},
        "weight": 0.00025, "target_tokens": 20_000_000,
    },
    {
        "name": "medmcqa", "category": "medical", "kind": "hf",
        "args": {"repo": "openlifescienceai/medmcqa", "split": "train"},
        "weight": 0.0001875, "target_tokens": 15_000_000,
    },
    {
        "name": "pubmedqa", "category": "medical", "kind": "hf",
        "args": {"repo": "qiaojin/PubMedQA", "subset": "pqa_artificial", "split": "train"},
        "weight": 0.0001875, "target_tokens": 15_000_000,
    },
    {
        "name": "wikipedia_medical", "category": "medical", "kind": "hf",
        "args": {"repo": "wikimedia/wikipedia", "subset": "20231101.en",
                 "split": "train", "streaming": True},
        "weight": 0.00625, "target_tokens": 500_000_000,
    },

    # ===== GENERAL =====
    {
        "name": "wikipedia_en", "category": "general", "kind": "hf",
        "args": {"repo": "wikimedia/wikipedia", "subset": "20231101.en",
                 "split": "train", "streaming": True},
        "weight": 0.05, "target_tokens": 4_000_000_000,
    },
    {
        "name": "c4", "category": "general", "kind": "hf",
        "args": {"repo": "allenai/c4", "subset": "en", "split": "train",
                 "streaming": True, "take": 50_000_000},
        "weight": 0.225, "target_tokens": 18_000_000_000,
    },
    {
        "name": "arxiv", "category": "general", "kind": "hf",
        "args": {"repo": "ccdv/arxiv-summarization", "split": "train",
                 "streaming": True},
        "weight": 0.0375, "target_tokens": 3_000_000_000,
    },
    {
        "name": "stackexchange", "category": "general", "kind": "hf",
        "args": {"repo": "HuggingFaceH4/stack-exchange-preferences",
                 "split": "train", "streaming": True},
        "weight": 0.00625, "target_tokens": 500_000_000,
    },
    # NOTE: bigcode/the-stack-smol is a GATED dataset. Even with HF_TOKEN set,
    # you must visit https://huggingface.co/datasets/bigcode/the-stack-smol and
    # accept its terms once before downloads will succeed.
    {
        "name": "code", "category": "general", "kind": "hf",
        "args": {"repo": "bigcode/the-stack-smol", "split": "train"},
        "weight": 0.0375, "target_tokens": 3_000_000_000,
    },
    {
        "name": "books", "category": "general", "kind": "hf",
        "args": {"repo": "emozilla/pg19", "split": "train", "streaming": True},
        "weight": 0.025, "target_tokens": 2_000_000_000,
    },
]


# ---------------------------------------------------------------------------
# Credentials -- read .env file then fall back to existing env vars.
# ---------------------------------------------------------------------------
def load_credentials(env_file: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (without overwriting)."""
    if env_file is None:
        candidates = [
            Path(__file__).parent / ".env",
            Path.cwd() / ".env",
        ]
        env_file = next((p for p in candidates if p.exists()), None)
    if env_file is None or not env_file.exists():
        return
    log.info("loading credentials from %s", env_file)
    for ln in env_file.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Atomic shard writer
# ---------------------------------------------------------------------------
def write_jsonl_gz(
    dest: Path,
    records: Iterable[dict],
    shard_size: int = SHARD_SIZE,
    prefix: str = "shard",
) -> int:
    """Stream records into <dest>/<prefix>_NNNNN.jsonl.gz with atomic rename.

    Resumes after the highest-numbered existing shard, so killing and
    re-running won't drop docs already on disk.
    """
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob(f"{prefix}_*.jsonl.gz"))
    shard_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
    total = 0
    fh = None
    tmp_path: Path | None = None
    final_path: Path | None = None
    in_shard = 0
    t0 = time.perf_counter()
    try:
        for rec in records:
            if fh is None:
                final_path = dest / f"{prefix}_{shard_idx:05d}.jsonl.gz"
                tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
                fh = gzip.open(tmp_path, "wt", encoding="utf-8")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            in_shard += 1
            if total % 5_000 == 0:
                dt = time.perf_counter() - t0
                rate = total / dt if dt else 0.0
                log.info("  %s: %s docs (%.0f/s)", dest.name, f"{total:,}", rate)
            if in_shard >= shard_size:
                fh.close()
                fh = None
                os.replace(tmp_path, final_path)
                tmp_path = final_path = None
                shard_idx += 1
                in_shard = 0
    finally:
        if fh is not None:
            fh.close()
            if tmp_path and final_path and in_shard > 0:
                os.replace(tmp_path, final_path)
            elif tmp_path and tmp_path.exists():
                tmp_path.unlink()
    return total


def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    """Tolerate truncation -- a SIGKILL'd shard is the most common corruption."""
    import zlib
    n = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                    n += 1
                except json.JSONDecodeError:
                    continue
    except (EOFError, OSError, zlib.error) as e:
        log.warning("truncated shard %s after %d docs (%s)", path.name, n, e.__class__.__name__)


def extract_text(ex: dict) -> str:
    for key in ("text", "content", "abstract", "article", "body"):
        v = ex.get(key)
        if isinstance(v, str) and v.strip():
            return v
    if "question" in ex and "long_answer" in ex:
        return f"Question: {ex['question']}\nAnswer: {ex['long_answer']}"
    if "instruction" in ex:
        out = ex.get("instruction", "")
        if ex.get("input"):
            out += "\n\n" + ex["input"]
        if ex.get("output"):
            out += "\n\n" + ex["output"]
        return out
    if "messages" in ex and isinstance(ex["messages"], list):
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in ex["messages"]
        )
    parts = [str(v) for v in ex.values() if isinstance(v, str)]
    return "\n".join(parts) if parts else ""


# ===========================================================================
# DOWNLOADERS
# ===========================================================================

def download_hf(
    dest: Path,
    *,
    repo: str,
    subset: str | None = None,
    split: str = "train",
    streaming: bool = False,
    take: int | None = None,
    max_docs: int | None = None,
    source_name: str = "",
    category: str = "",
    **_: Any,
) -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from datasets import load_dataset  # type: ignore
    log.info("hf: %s subset=%s split=%s streaming=%s", repo, subset, split, streaming)
    ds = (
        load_dataset(repo, subset, split=split, streaming=streaming)
        if subset
        else load_dataset(repo, split=split, streaming=streaming)
    )
    cap = min(x for x in (take, max_docs) if x is not None) if (take or max_docs) else None

    def _records() -> Iterator[dict]:
        n = 0
        for ex in ds:
            text = extract_text(ex)
            if not text or not text.strip():
                continue
            yield {"text": text, "source": source_name or repo, "category": category}
            n += 1
            if cap and n >= cap:
                break

    return write_jsonl_gz(dest, _records())


def download_pubmed(
    dest: Path,
    *,
    years: list[int] | None = None,
    host: str = "ftp.ncbi.nlm.nih.gov",
    base_path: str = "/pubmed/baseline/",
    max_files: int | None = None,
    max_docs: int | None = None,
    source_name: str = "pubmed",
    category: str = "medical",
    **_: Any,
) -> int:
    raw_xml = dest / "_raw_xml"
    raw_xml.mkdir(parents=True, exist_ok=True)
    # NCBI hosts the same archive over HTTPS at the same path. HTTPS avoids the
    # passive-FTP control/data split that drops through NATs and firewalls.
    base_url = f"https://{host}{base_path}"
    log.info("pubmed: https %s", base_url)
    req = urllib.request.Request(base_url, headers={"User-Agent": "medllm-data/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        index_html = resp.read().decode("utf-8", errors="ignore")
    files = sorted(set(re.findall(r'href="(pubmed\d+n\d+\.xml\.gz)"', index_html)))
    # PubMed baselines are PMID-sorted (low file number = oldest articles).
    # When max_files is set (smoke test), prefer the newest files so the
    # year filter actually matches something.
    if max_files:
        files = files[-max_files:]
    log.info("pubmed: %d files", len(files))
    for fname in files:
        target = raw_xml / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        tmp = target.with_suffix(target.suffix + ".tmp")
        url = base_url + fname
        log.info("pubmed: GET %s", fname)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "medllm-data/0.1"})
                with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                break
            except Exception as e:
                log.warning("pubmed: attempt %d for %s failed (%s)", attempt + 1, fname, e)
                if tmp.exists():
                    tmp.unlink()
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        os.replace(tmp, target)

    def _records() -> Iterator[dict]:
        n = 0
        for xml_gz in sorted(raw_xml.glob("*.xml.gz")):
            try:
                with gzip.open(xml_gz, "rb") as f:
                    tree = ET.parse(f)
            except (ET.ParseError, OSError) as e:
                log.warning("pubmed: skipping %s (%s)", xml_gz.name, e)
                continue
            for art in tree.iterfind(".//PubmedArticle"):
                yr_node = art.find(".//PubDate/Year")
                year = (
                    int(yr_node.text)
                    if yr_node is not None and yr_node.text and yr_node.text.isdigit()
                    else None
                )
                if years and year not in years:
                    continue
                title_el = art.find(".//ArticleTitle")
                abs_els = art.findall(".//Abstract/AbstractText")
                title = "".join(title_el.itertext()) if title_el is not None else ""
                abstract = "\n".join("".join(a.itertext()) for a in abs_els)
                if not abstract.strip():
                    continue
                yield {
                    "text": f"{title}\n\n{abstract}".strip(),
                    "source": source_name,
                    "category": category,
                    "year": year,
                }
                n += 1
                if max_docs and n >= max_docs:
                    return

    return write_jsonl_gz(dest, _records())


def download_ctgov(
    dest: Path,
    *,
    fields: list[str] | None = None,
    page_size: int = 1000,
    max_pages: int | None = None,
    max_docs: int | None = None,
    source_name: str = "clinicaltrials",
    category: str = "medical",
    **_: Any,
) -> int:
    fields = fields or ["BriefSummary", "DetailedDescription", "EligibilityCriteria"]
    base = "https://clinicaltrials.gov/api/v2/studies"

    def _records() -> Iterator[dict]:
        page_token: str | None = None
        pages = 0
        n = 0
        while True:
            params = [f"pageSize={page_size}", "format=json"]
            if page_token:
                params.append(f"pageToken={page_token}")
            url = f"{base}?{'&'.join(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "medllm-data/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                log.error("ctgov: %s -- backoff 5s", e)
                time.sleep(5)
                continue
            for s in data.get("studies", []):
                ps = s.get("protocolSection", {})
                desc = ps.get("descriptionModule", {})
                elig = ps.get("eligibilityModule", {})
                buf = []
                if "BriefSummary" in fields and desc.get("briefSummary"):
                    buf.append(desc["briefSummary"])
                if "DetailedDescription" in fields and desc.get("detailedDescription"):
                    buf.append(desc["detailedDescription"])
                if "EligibilityCriteria" in fields and elig.get("eligibilityCriteria"):
                    buf.append("Eligibility: " + elig["eligibilityCriteria"])
                if buf:
                    yield {
                        "text": "\n\n".join(buf),
                        "source": source_name,
                        "category": category,
                    }
                    n += 1
                    if max_docs and n >= max_docs:
                        return
            page_token = data.get("nextPageToken")
            pages += 1
            if not page_token or (max_pages and pages >= max_pages):
                return

    return write_jsonl_gz(dest, _records())


DOWNLOADERS: dict[str, Callable[..., int]] = {
    "hf": download_hf,
    "pubmed_ftp": download_pubmed,
    "ctgov": download_ctgov,
}


# ===========================================================================
# QUALITY FILTERS  (mirrors src/llm/data/filters.py)
# ===========================================================================
_BULLET_RE = re.compile(r"^\s*([-*•·]|\d+[.)])\s+", re.MULTILINE)
_WORD_RE = re.compile(r"\w+")
_NON_ALNUM_RE = re.compile(r"[^\w\s]")


@dataclass
class QualityRules:
    min_chars: int = 200
    max_chars: int = 200_000
    mean_word_len_range: tuple[int, int] = (3, 12)
    bullet_ratio_max: float = 0.95
    symbol_to_word_ratio_max: float = 0.3
    repeated_line_ratio_max: float = 0.5


def _mean_word_length(text: str) -> float:
    words = _WORD_RE.findall(text)
    return statistics.fmean(len(w) for w in words) if words else 0.0


def _bullet_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if _BULLET_RE.match(ln)) / len(lines)


def _symbol_to_word_ratio(text: str) -> float:
    words = _WORD_RE.findall(text)
    syms = _NON_ALNUM_RE.findall(text)
    return (len(syms) / len(words)) if words else 1.0


def _repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    repeats = sum(c for c in seen.values() if c > 1)
    return repeats / len(lines)


def quality_filter(text: str, rules: QualityRules) -> bool:
    n = len(text)
    if n < rules.min_chars or n > rules.max_chars:
        return False
    mwl = _mean_word_length(text)
    lo, hi = rules.mean_word_len_range
    if not (lo <= mwl <= hi):
        return False
    if _bullet_ratio(text) > rules.bullet_ratio_max:
        return False
    if _symbol_to_word_ratio(text) > rules.symbol_to_word_ratio_max:
        return False
    if _repeated_line_ratio(text) > rules.repeated_line_ratio_max:
        return False
    return True


def language_filter(text: str, allowed: tuple[str, ...] = ("en",), min_score: float = 0.7) -> bool:
    """Optional: requires `langdetect`. If not installed, passes through."""
    try:
        from langdetect import DetectorFactory, detect_langs  # type: ignore
        DetectorFactory.seed = 0
        for c in detect_langs(text[:2000]):
            if c.lang in allowed and c.prob >= min_score:
                return True
        return False
    except Exception:
        return True


# ===========================================================================
# PHI / PII REDACTION  (mirrors src/llm/safety/phi_redaction.py -- HIPAA Safe Harbor)
# Patterns are intentionally CONSERVATIVE: over-redact, never leak.
# Order matters: SSN before phone, etc.
# ===========================================================================
_PHI_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:SSN|Social Security)[:\s#]+\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", re.I), "[REDACTED_SSN]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\bhttps?://[^\s]+"), "[REDACTED_URL]"),
    (re.compile(r"\bwww\.[^\s]+"), "[REDACTED_URL]"),
    (re.compile(r"\b(?:MRN|Medical Record(?: Number)?|Patient ID|Chart)[\s:#]+\d{4,12}\b", re.I), "[REDACTED_MRN]"),
    (re.compile(r"\b(?:MRN|Medical Record|Patient ID|Chart)[\s:#]+[A-Z]{1,4}-?\d{4,12}\b", re.I), "[REDACTED_MRN]"),
    (re.compile(r"\b(?:Member|Subscriber|Policy|Group|Insurance)\s*(?:ID|Number|#)[:\s]+[A-Z0-9-]{6,15}\b", re.I), "[REDACTED_HEALTH_PLAN_ID]"),
    (re.compile(r"\b(?:Account|Acct|Bill|Invoice)\s*(?:Number|#)[:\s]+[A-Z0-9-]{6,15}\b", re.I), "[REDACTED_ACCOUNT]"),
    (re.compile(r"\b(?:License|Licence|Certificate|Cert|DEA)\s*(?:Number|#)[:\s]+[A-Z0-9-]{5,15}\b", re.I), "[REDACTED_LICENSE]"),
    (re.compile(r"\b(?:Device|Serial|S/N)\s*(?:Number|#)[:\s]+[A-Z0-9-]{6,20}\b", re.I), "[REDACTED_DEVICE_ID]"),
    (re.compile(r"\b(?:License Plate|VIN|Vehicle ID)[:\s]+[A-Z0-9-]{5,17}\b", re.I), "[REDACTED_VEHICLE]"),
    (re.compile(r"\b(?:phone|tel|telephone|fax|cell|mobile)[:\s]+(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", re.I), "[REDACTED_PHONE]"),
    (re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:DOB|Date of Birth|Born)[:\s]+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.I), "[REDACTED_DATE]"),
    (re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}\b"), "[REDACTED_DATE]"),
    (re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b", re.I), "[REDACTED_DATE]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    (re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"), "[REDACTED_IP]"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "[REDACTED_UUID]"),
    # NOTE: name patterns rely on capitalization -- DO NOT add IGNORECASE here.
    (re.compile(r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor)\.?\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\b"), "[REDACTED_NAME]"),
    (re.compile(r"\b(?:Patient|Subject|Pt)\s+(?:[Nn]ame[:\s]+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"), "[REDACTED_NAME]"),
    (re.compile(r"\b(?:Name|name)[:\s]+[A-Z][a-z]+\s+[A-Z][a-z]+\b"), "[REDACTED_NAME]"),
    (re.compile(r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Circle|Cir)\b"), "[REDACTED_ADDRESS]"),
    (re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"), "[REDACTED_ADDRESS]"),
    (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[REDACTED_ZIP]"),
]


def phi_redact(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, repl in _PHI_PATTERNS:
        out = pat.sub(repl, out)
    return out


# ===========================================================================
# DEDUPLICATION
# ===========================================================================
def sha256_dedup(records: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()
    for r in records:
        h = hashlib.sha256(r["text"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        yield r


_TOKEN_RE = re.compile(r"\w+")


def _shingles(text: str, n: int = 5) -> set[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i: i + n]) for i in range(len(toks) - n + 1)}


def _minhash_sig(shingles: set[str], num_perm: int) -> list[int]:
    if not shingles:
        return [0] * num_perm
    sig = [(1 << 64) - 1] * num_perm
    for sh in shingles:
        base = int(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(num_perm):
            h = (base * (i * 2654435761 + 1)) & ((1 << 64) - 1)
            if h < sig[i]:
                sig[i] = h
    return sig


def _optimal_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    best = (1, num_perm)
    best_diff = 1.0
    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        s = (1.0 / b) ** (1.0 / r)
        d = abs(s - threshold)
        if d < best_diff:
            best_diff = d
            best = (b, r)
    return best


def minhash_dedup(
    records: Iterable[dict],
    num_perm: int = 128,
    threshold: float = 0.85,
    ngram: int = 5,
) -> Iterator[dict]:
    bands, rows = _optimal_bands(num_perm, threshold)
    buckets: list[dict[tuple[int, ...], list[int]]] = [defaultdict(list) for _ in range(bands)]
    sigs: list[list[int]] = []
    for r in records:
        sig = _minhash_sig(_shingles(r["text"], ngram), num_perm)
        bnds = [tuple(sig[b * rows: (b + 1) * rows]) for b in range(bands)]
        dup = False
        for bi, band in enumerate(bnds):
            for other in buckets[bi][band]:
                eq = sum(1 for a, b in zip(sig, sigs[other]) if a == b)
                if (eq / num_perm) >= threshold:
                    dup = True
                    break
            if dup:
                break
        if dup:
            continue
        idx = len(sigs)
        sigs.append(sig)
        for bi, band in enumerate(bnds):
            buckets[bi][band].append(idx)
        yield r


# ===========================================================================
# PARALLEL FILTER WORKER  (one process per raw shard)
# ===========================================================================
def _filter_shard_worker(
    shard_path: str,
    out_path: str,
    rules_dict: dict,
    do_lang: bool,
    do_pii: bool,
) -> tuple[str, int, int]:
    src = Path(shard_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    rules = QualityRules(**rules_dict)
    seen = kept = 0
    sys.stderr.write(f"[filter] {src.name}: started\n")
    sys.stderr.flush()
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for rec in iter_jsonl_gz(src):
            seen += 1
            text = rec.get("text", "")
            if not text:
                continue
            if not quality_filter(text, rules):
                continue
            if do_lang and not language_filter(text):
                continue
            if do_pii:
                rec = {**rec, "text": phi_redact(text)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if seen % 10_000 == 0:
                sys.stderr.write(f"[filter] {src.name}: seen={seen:,} kept={kept:,}\n")
                sys.stderr.flush()
    os.replace(tmp, out)
    sys.stderr.write(f"[filter] {src.name}: done seen={seen:,} kept={kept:,}\n")
    sys.stderr.flush()
    return shard_path, seen, kept


# ===========================================================================
# DRIVER -- per-source: download -> filter -> dedup -> redact -> write cleaned
# ===========================================================================
def run_download(src: dict, src_dir: Path, max_docs: int | None,
                 max_files: int | None = None) -> None:
    raw_dir = src_dir / "raw"
    marker = src_dir / "_DOWNLOAD_DONE"
    if marker.exists():
        log.info("[%s] _DOWNLOAD_DONE present, skipping download", src["name"])
        return
    fn = DOWNLOADERS[src["kind"]]
    args = dict(src["args"])
    if max_docs is not None:
        args["max_docs"] = max_docs
    if max_files is not None:
        args["max_files"] = max_files
    args["source_name"] = src["name"]
    args["category"] = src["category"]
    log.info("=== download %s/%s ===", src["category"], src["name"])
    t0 = time.perf_counter()
    kept = fn(raw_dir, **args)
    dt = time.perf_counter() - t0
    log.info("[%s] downloaded %s docs in %.1fs", src["name"], f"{kept:,}", dt)
    marker.write_text(json.dumps({
        "docs": kept, "seconds": dt,
        "category": src["category"], "weight": src["weight"],
        "target_tokens": src["target_tokens"],
    }))


def run_clean(
    src: dict,
    src_dir: Path,
    rules: QualityRules,
    do_lang: bool,
    do_pii: bool,
    do_minhash: bool,
    minhash_cfg: dict,
    num_workers: int,
) -> None:
    raw_dir = src_dir / "raw"
    cleaned_dir = src_dir / "cleaned"
    done_marker = src_dir / "_CLEAN_DONE"
    if done_marker.exists():
        log.info("[%s] _CLEAN_DONE present, skipping clean", src["name"])
        return
    raw_shards = sorted(raw_dir.glob("shard_*.jsonl.gz"))
    if not raw_shards:
        log.warning("[%s] no raw shards in %s -- run download first", src["name"], raw_dir)
        return

    # wipe stale partials
    if cleaned_dir.exists():
        for stale in list(cleaned_dir.glob("*.jsonl.gz*")):
            log.warning("[%s] removing stale %s", src["name"], stale.name)
            stale.unlink()

    # Stage 1: parallel filter
    filt_dir = src_dir / "_filtered"
    filt_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for shard in raw_shards:
        out_p = filt_dir / f"filt_{shard.stem}.jsonl.gz"
        if out_p.exists():
            log.info("[%s] reusing filtered %s", src["name"], out_p.name)
            continue
        work.append((str(shard), str(out_p), rules.__dict__, do_lang, do_pii))

    if work:
        n_proc = max(1, min(len(work), num_workers))
        log.info("[%s] filtering %d shards with %d workers", src["name"], len(work), n_proc)
        with ProcessPoolExecutor(max_workers=n_proc) as pool:
            futs = [pool.submit(_filter_shard_worker, *w) for w in work]
            for fut in as_completed(futs):
                p, seen, kept = fut.result()
                log.info("[%s] %s seen=%d kept=%d", src["name"], Path(p).name, seen, kept)

    filt_shards = sorted(filt_dir.glob("*.jsonl.gz"))
    if not filt_shards:
        raise RuntimeError(f"[{src['name']}] filter produced no output")

    # Stage 2: dedup (sha256 + optional minhash)
    def _merged() -> Iterator[dict]:
        for fs in filt_shards:
            yield from iter_jsonl_gz(fs)

    def _tick(it: Iterable[dict], stage: str, every: int = 10_000) -> Iterator[dict]:
        n = 0
        t0 = time.perf_counter()
        for x in it:
            yield x
            n += 1
            if n % every == 0:
                dt = time.perf_counter() - t0
                log.info("[%s] %s: %d docs in %.1fs (%.0f/s)",
                         src["name"], stage, n, dt, n / dt if dt else 0)

    pipeline: Iterable[dict] = sha256_dedup(_tick(_merged(), "sha256"))
    if do_minhash:
        pipeline = minhash_dedup(
            _tick(pipeline, "minhash"),
            num_perm=int(minhash_cfg.get("num_perm", 128)),
            threshold=float(minhash_cfg.get("threshold", 0.85)),
            ngram=int(minhash_cfg.get("ngram_size", 5)),
        )

    kept = write_jsonl_gz(cleaned_dir, pipeline, prefix="cleaned")
    done_marker.write_text(json.dumps({
        "kept": kept,
        "category": src["category"], "weight": src["weight"],
        "target_tokens": src["target_tokens"],
    }))
    log.info("[%s] CLEAN DONE: kept=%s docs", src["name"], f"{kept:,}")

    # cleanup intermediate
    for fp in filt_dir.glob("*.jsonl.gz"):
        fp.unlink()
    if filt_dir.exists() and not any(filt_dir.iterdir()):
        filt_dir.rmdir()


# ===========================================================================
# YAML config emitter for the VM side
# ===========================================================================
def emit_vm_config(out_dir: Path, sources: list[dict]) -> str:
    lines = [
        "# Auto-emitted by prepare_data_local.py --emit-config",
        "# Drop this into configs/data/ on the VM and point training at it.",
        "data:",
        "  format: streaming",
        "  shard_glob: ./data/processed/*/*.bin",
        "  index_glob: ./data/processed/*/*.idx",
        "  tokenizer: ./checkpoints/tokenizer",
        "  sequence_length: 8192",
        "  packing: true",
        "  pack_strategy: best_fit_decreasing",
        "  add_bos: true",
        "  add_eos: true",
        "  target_tokens: 80_000_000_000",
        "  sources:",
    ]
    for s in sources:
        cleaned_path = f"./data/local_prepared/{s['category']}/{s['name']}/cleaned"
        proc_path = f"./data/processed/{s['name']}"
        lines += [
            f"    - name: {s['name']}",
            f"      category: {s['category']}",
            f"      raw_dir: {cleaned_path}",
            f"      processed_dir: {proc_path}",
            f"      target_tokens: {s['target_tokens']}",
            f"      weight: {s['weight']}",
        ]
    lines += [
        "  preprocessing:",
        "    min_doc_chars: 200",
        "    max_doc_chars: 200_000",
        "    pii_redact: false        # already done locally",
        "    medical_phi_redact: false  # already done locally",
        "    dedup:",
        "      exact_sha256: false    # already done locally",
        "      minhash: { enabled: false }  # already done locally",
    ]
    return "\n".join(lines) + "\n"


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="./data",
                   help="root output dir (creates <out>/<category>/<source>/...)")
    p.add_argument("--source", action="append", default=None,
                   help="restrict to this source name (repeatable)")
    p.add_argument("--category", choices=["medical", "general"], default=None,
                   help="restrict to this category")
    p.add_argument("--max-docs", type=int, default=None,
                   help="cap docs per source (smoke test)")
    p.add_argument("--max-files", type=int, default=None,
                   help="cap raw files per source (only used by pubmed FTP "
                        "downloader; useful for smoke tests)")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel filter workers (0 = auto = CPU-2)")
    p.add_argument("--download-only", action="store_true",
                   help="only run download stage")
    p.add_argument("--clean-only", action="store_true",
                   help="only run filter+dedup+redact stage (raw shards must exist)")
    p.add_argument("--no-pii", action="store_true",
                   help="disable PHI/PII redaction (NOT recommended for medical)")
    p.add_argument("--no-minhash", action="store_true",
                   help="skip MinHash fuzzy dedup (faster; only sha256 dedup runs)")
    p.add_argument("--no-langid", action="store_true",
                   help="skip language id filter")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (overrides env)")
    p.add_argument("--credentials-file", default=None,
                   help="path to .env file with HF_TOKEN= etc.")
    p.add_argument("--list", action="store_true", help="list sources and exit")
    p.add_argument("--emit-config", action="store_true",
                   help="print a YAML config for the VM and exit")
    args = p.parse_args()

    # ---- credentials ----
    load_credentials(Path(args.credentials_file) if args.credentials_file else None)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    # ---- listing / config emission ----
    if args.list:
        print(f"{'name':22s}  {'category':10s}  {'kind':12s}  weight   target_tokens")
        for s in SOURCES:
            print(f"  {s['name']:20s}  {s['category']:10s}  {s['kind']:12s}  "
                  f"{s['weight']:.5f}  {s['target_tokens']:>14,}")
        return 0

    out_dir = Path(args.out_dir)
    if args.emit_config:
        sys.stdout.write(emit_vm_config(out_dir, SOURCES))
        return 0

    # ---- select sources ----
    selected = SOURCES
    if args.category:
        selected = [s for s in selected if s["category"] == args.category]
    if args.source:
        wanted = set(args.source)
        selected = [s for s in selected if s["name"] in wanted]
        missing = wanted - {s["name"] for s in selected}
        if missing:
            log.error("unknown source(s): %s", sorted(missing))
            return 2
    if not selected:
        log.error("no sources match the filter")
        return 2

    # ---- preprocessing config ----
    rules = QualityRules()
    do_pii = not args.no_pii
    do_lang = not args.no_langid
    do_minhash = not args.no_minhash
    minhash_cfg = {"num_perm": 128, "threshold": 0.85, "ngram_size": 5}
    auto_workers = max(1, (os.cpu_count() or 4) - 2)
    num_workers = auto_workers if args.workers <= 0 else args.workers
    log.info("settings: pii=%s langid=%s minhash=%s workers=%d",
             do_pii, do_lang, do_minhash, num_workers)

    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    summaries: list[dict] = []

    for src in selected:
        src_dir = out_dir / src["category"] / src["name"]
        src_dir.mkdir(parents=True, exist_ok=True)
        # write a small _meta.json so the folder is self-describing
        (src_dir / "_meta.json").write_text(json.dumps({
            "name": src["name"], "category": src["category"], "kind": src["kind"],
            "weight": src["weight"], "target_tokens": src["target_tokens"],
            "args": src["args"],
        }, indent=2))

        try:
            if not args.clean_only:
                run_download(src, src_dir, args.max_docs, args.max_files)
            if not args.download_only:
                run_clean(src, src_dir, rules, do_lang, do_pii,
                          do_minhash, minhash_cfg, num_workers)
        except KeyboardInterrupt:
            log.error("interrupted")
            return 130
        except Exception as e:
            log.exception("[%s] failed: %s", src["name"], e)
            failures.append((src["name"], repr(e)))
            continue

        cleaned_dir = src_dir / "cleaned"
        n_clean = sum(1 for _ in cleaned_dir.glob("cleaned_*.jsonl.gz")) if cleaned_dir.exists() else 0
        summaries.append({
            "source": src["name"], "category": src["category"],
            "cleaned_shards": n_clean,
        })

    print(json.dumps({"summary": summaries, "failures": failures}, indent=2))
    log.info("done. sync:  rsync -avh %s/ user@vm:/path/data/local_prepared/", out_dir)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
