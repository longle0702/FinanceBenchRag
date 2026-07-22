"""
Step 1: Automate the PDF Retrieval.

Loads data/financebench_document_information.jsonl, extracts the unique
(doc_name, doc_link) pairs, and downloads each filing PDF into a local
directory (default: data/raw_pdfs/). Downloads run concurrently (default: 4
workers) and show an overall progress bar across documents plus one
byte-level progress bar per in-flight download.

Usage:
    python scripts/download_pdfs.py
    python scripts/download_pdfs.py --workers 4
    python scripts/download_pdfs.py --limit 5 --overwrite
    python scripts/download_pdfs.py --user-agent "YourName your@email.com"
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "financebench_document_information.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw_pdfs"

# SEC EDGAR requires a descriptive User-Agent identifying the requester
# (see https://www.sec.gov/os/webmaster-faq#developers). Override with --user-agent.
DEFAULT_USER_AGENT = "FinanceBenchRag/1.0 (duongnt.mindx@gmail.com)"


class DownloadTimeout(Exception):
    """Raised when a single file's total download time exceeds the allowed budget."""


# Each worker thread gets its own requests.Session and a fixed tqdm bar position
# (assigned once, on first use) so concurrent downloads don't fight over the same line.
# Position-based multi-bar rendering needs ANSI cursor control, which only works on a
# real interactive terminal. To stay visible even when output is redirected/logged
# (e.g. a background task), we also track the set of in-flight downloads and reflect
# it in the single overall bar's postfix, plus log explicit start/finish lines.
_thread_local = threading.local()
_position_lock = threading.Lock()
_next_position = [1]

_active_lock = threading.Lock()
_active_downloads: dict[str, None] = {}


def _get_thread_session(user_agent: str, retries: int, backoff_factor: float) -> tuple[requests.Session, int]:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = build_session(user_agent, retries, backoff_factor)
        _thread_local.session = session
        with _position_lock:
            _thread_local.position = _next_position[0]
            _next_position[0] += 1
    return session, _thread_local.position


def _set_active(overall_bar: tqdm, doc_name: str, failed_count: int, active: bool) -> None:
    with _active_lock:
        if active:
            _active_downloads[doc_name] = None
        else:
            _active_downloads.pop(doc_name, None)
        names = ", ".join(_active_downloads) if _active_downloads else "-"
    overall_bar.set_postfix_str(f"failed={failed_count} downloading=[{names}]")


def load_documents(jsonl_path: Path) -> list[dict]:
    """Read the catalog JSONL and return unique (doc_name, doc_link) pairs, sorted by doc_name."""
    unique: dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_name = record.get("doc_name")
            doc_link = record.get("doc_link")
            if doc_name and doc_link:
                unique[doc_name] = doc_link
    return [{"doc_name": name, "doc_link": unique[name]} for name in sorted(unique)]


def build_session(user_agent: str, retries: int, backoff_factor: float) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    })
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_pdf(
    session: requests.Session,
    doc_name: str,
    url: str,
    out_dir: Path,
    timeout: float,
    overwrite: bool,
    max_download_time: float,
    position: int = 1,
) -> tuple[str, str | None]:
    """Download a single PDF with a byte-level progress bar. Returns (status, error)."""
    dest = out_dir / f"{doc_name}.pdf"
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return "skipped", None

    tmp_path = dest.with_suffix(".part")
    start = time.monotonic()
    try:
        # timeout is (connect_timeout, read_timeout): read_timeout only bounds the
        # gap between individual chunks, not the whole download, so a server that
        # trickles bytes slowly enough can otherwise stall forever.
        with session.get(url, stream=True, timeout=(min(timeout, 10.0), timeout)) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            with open(tmp_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=doc_name,
                leave=False,
                position=position,
            ) as file_bar:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if time.monotonic() - start > max_download_time:
                        raise DownloadTimeout(f"exceeded {max_download_time:.0f}s total download budget")
                    if chunk:
                        f.write(chunk)
                        file_bar.update(len(chunk))
        tmp_path.replace(dest)
        return "downloaded", None
    except (requests.RequestException, DownloadTimeout) as exc:
        tmp_path.unlink(missing_ok=True)
        return "failed", str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FinanceBench source PDFs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to financebench_document_information.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save PDFs into")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent header sent with each request")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument("--max-download-time", type=float, default=120.0, help="Hard cap (seconds) on a single file's total download time before it's aborted and marked failed")
    parser.add_argument("--retries", type=int, default=5, help="Automatic retries per file on failure")
    parser.add_argument("--backoff-factor", type=float, default=1.5, help="Exponential backoff factor between retries")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay (seconds) each worker waits after finishing a request, to stay polite with SEC EDGAR rate limits")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent downloads (SEC EDGAR allows up to 10 req/s; 4 is a safe default)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--limit", type=int, default=None, help="Only download the first N documents (useful for testing)")
    parser.add_argument("--report-path", type=Path, default=None, help="Where to write the CSV status report (default: <output-dir>/download_report.csv)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or (args.output_dir / "download_report.csv")

    documents = load_documents(args.input)
    if args.limit:
        documents = documents[: args.limit]

    failed_count = [0]  # mutated under _active_lock, read from any thread

    # position=0 is reserved for the overall bar; worker bars start at position 1.
    overall_bar = tqdm(total=len(documents), unit="doc", desc="Overall progress", position=0)

    def _download_one(doc: dict) -> tuple[dict, str, str | None]:
        doc_name = doc["doc_name"]
        session, position = _get_thread_session(args.user_agent, args.retries, args.backoff_factor)
        _set_active(overall_bar, doc_name, failed_count[0], active=True)
        tqdm.write(f"-> starting {doc_name}")
        status, error = download_pdf(
            session, doc_name, doc["doc_link"], args.output_dir, args.timeout, args.overwrite,
            args.max_download_time, position,
        )
        if status == "failed":
            with _active_lock:
                failed_count[0] += 1
        _set_active(overall_bar, doc_name, failed_count[0], active=False)
        mark = {"downloaded": "OK", "skipped": "--", "failed": "XX"}[status]
        tqdm.write(f"<- [{mark}] {status:<10} {doc_name}" + (f" ({error})" if error else ""))
        time.sleep(args.delay)
        return doc, status, error

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    records: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download_one, doc) for doc in documents]
        for future in as_completed(futures):
            doc, status, error = future.result()
            counts[status] += 1
            records.append({
                "doc_name": doc["doc_name"],
                "doc_link": doc["doc_link"],
                "status": status,
                "error": error or "",
            })
            overall_bar.update(1)
    overall_bar.close()

    # Restore stable ordering (as_completed finishes in whatever order threads land in).
    records.sort(key=lambda r: r["doc_name"])

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_name", "doc_link", "status", "error"])
        writer.writeheader()
        writer.writerows(records)

    working = counts["downloaded"] + counts["skipped"]
    total = len(records)
    print(
        f"\n{working}/{total} links work "
        f"(downloaded: {counts['downloaded']}, already present: {counts['skipped']}), "
        f"{counts['failed']}/{total} failed."
    )
    print(f"Full report written to {report_path}")

    failures = [r for r in records if r["status"] == "failed"]
    if failures:
        print("\nFailed links:")
        for r in failures:
            print(f"  - {r['doc_name']}: {r['error']} ({r['doc_link']})")


if __name__ == "__main__":
    main()
