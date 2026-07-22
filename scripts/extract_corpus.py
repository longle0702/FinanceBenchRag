"""
Step 2: Build Your Knowledge Base (Ingestion).

Processes downloaded PDFs from data/raw_pdfs/ into three interleaved,
doc-ordered chunk streams (narrative text, financial tables, glossary/defined
terms) plus a per-doc and corpus-level glossary lookup, ready for the
embedding/indexing stage.

Usage:
    python scripts/extract_corpus.py
    python scripts/extract_corpus.py --chunk-size 500 --chunk-overlap 75
    python scripts/extract_corpus.py --limit 5
    python scripts/extract_corpus.py --doc-name 3M_2018_10K --overwrite
    python scripts/extract_corpus.py --table-backend pdfplumber
    python scripts/extract_corpus.py --table-format html
    python scripts/extract_corpus.py --no-table-images
    python scripts/extract_corpus.py --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from extract import narrative
from extract import pdf_io
from extract import tables as tables_mod
from extract.config import ExtractionConfig
from extract.glossary import build_global_rollup, extract_all_entries
from extract.narrative import ParagraphUnit
from extract.schema import Chunk, GlossaryEntry, link_neighbors, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "data" / "financebench_document_information.jsonl"
DEFAULT_PDF_DIR = REPO_ROOT / "data" / "raw_pdfs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"


def load_catalog(path: Path) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_name = record.get("doc_name")
            if doc_name:
                catalog[doc_name] = record
    return catalog


def make_chunk_id(doc_name: str, page_num: int, chunk_index: int) -> str:
    return f"{doc_name}__p{page_num:04d}__c{chunk_index:04d}"


def process_document(
    pdf_path: Path,
    catalog_entry: dict,
    config: ExtractionConfig,
    table_backend: str,
    output_dir: Path,
) -> tuple[list[Chunk], list[GlossaryEntry], dict]:
    doc_name = catalog_entry["doc_name"]
    doc_type = catalog_entry.get("doc_type", "unknown")
    company = catalog_entry.get("company", "unknown")
    doc_period = str(catalog_entry.get("doc_period", "unknown"))

    doc = pdf_io.open_document(pdf_path)
    try:
        sample_pages = list(doc)[: min(5, doc.page_count)]
        avg_chars = sum(pdf_io.extractable_char_count(p) for p in sample_pages) / max(1, len(sample_pages))
        if avg_chars < config.min_chars_per_page:
            return [], [], {"status": "likely_scanned_or_image", "n_pages": doc.page_count}

        boilerplate = pdf_io.compute_boilerplate_lines(doc, config)
        median_size = pdf_io.compute_median_body_font_size(doc)
        toc_map = pdf_io.build_toc_section_map(doc)

        pdfplumber_doc = None
        if table_backend == "pdfplumber":
            import pdfplumber
            pdfplumber_doc = pdfplumber.open(pdf_path)

        all_chunks: list[Chunk] = []
        all_paragraphs: list[ParagraphUnit] = []
        narrative_buffer: list[ParagraphUnit] = []
        chunk_index = 0
        table_counter = 0
        current_section_id: str | None = None
        current_section_title: str | None = None

        def flush_narrative() -> None:
            nonlocal chunk_index, narrative_buffer
            if not narrative_buffer:
                return
            for group in narrative.pack_paragraphs(narrative_buffer, config):
                text = "\n\n".join(p.text for p in group)
                all_chunks.append(Chunk(
                    chunk_id=make_chunk_id(doc_name, group[0].page_num, chunk_index),
                    doc_name=doc_name, doc_type=doc_type, company=company, doc_period=doc_period,
                    chunk_type="narrative",
                    section=group[0].section_title, section_id=group[0].section_id,
                    page_start=group[0].page_num, page_end=group[-1].page_num,
                    chunk_index=chunk_index, n_tokens=narrative.count_tokens(text), text=text,
                    source={"extractor": "pymupdf.get_text"},
                ))
                chunk_index += 1
            narrative_buffer = []

        try:
            for page in doc:
                pdfplumber_page = pdfplumber_doc.pages[page.number] if pdfplumber_doc else None
                blocks = pdf_io.get_page_blocks(page)
                page_tables = [
                    t for t in pdf_io.get_page_tables(page, table_backend, pdfplumber_page)
                    if tables_mod.is_plausible_table(t)
                ]

                if toc_map and page.number in toc_map:
                    current_section_id, current_section_title = toc_map[page.number]

                items: list[tuple[str, object, float]] = []
                for block in blocks:
                    if pdf_io.block_in_tables(block, page_tables, config):
                        continue
                    items.append(("block", block, block.bbox[1]))
                for region in page_tables:
                    items.append(("table", region, region.bbox[1]))
                items.sort(key=lambda x: x[2])

                last_paragraph_text: str | None = None
                for kind, obj, _y in items:
                    if kind == "block":
                        if not toc_map:
                            for line in obj.lines:
                                heading = pdf_io.detect_heading_line(line, median_size, config)
                                if heading:
                                    if heading[0] != current_section_id:
                                        flush_narrative()
                                    current_section_id, current_section_title = heading
                                    break
                        for text in narrative.join_block_paragraphs(obj, boilerplate):
                            unit = ParagraphUnit(
                                text=text, page_num=page.number,
                                section_id=current_section_id, section_title=current_section_title,
                                tokens=narrative.count_tokens(text),
                            )
                            narrative_buffer.append(unit)
                            all_paragraphs.append(unit)
                            last_paragraph_text = text
                    else:
                        flush_narrative()
                        table_counter += 1
                        table_id = f"{doc_name}__table{table_counter:03d}"
                        image_path = None
                        if config.table_images_enabled:
                            image_file = (
                                output_dir / "table_images" / doc_name
                                / f"{table_id}.{config.table_image_format}"
                            )
                            pdf_io.save_table_crop(
                                page, obj.bbox, image_file,
                                dpi=config.table_image_dpi,
                                image_format=config.table_image_format,
                                padding_pt=config.table_image_padding_pt,
                            )
                            image_path = image_file.relative_to(REPO_ROOT).as_posix()
                        drafts = tables_mod.build_table_chunks(
                            obj, last_paragraph_text, table_id, company, doc_period, config, image_path,
                        )
                        for draft in drafts:
                            all_chunks.append(Chunk(
                                chunk_id=make_chunk_id(doc_name, draft.page_num, chunk_index),
                                doc_name=doc_name, doc_type=doc_type, company=company, doc_period=doc_period,
                                chunk_type="table",
                                section=current_section_title, section_id=current_section_id,
                                page_start=draft.page_num, page_end=draft.page_num,
                                chunk_index=chunk_index, n_tokens=narrative.count_tokens(draft.text),
                                text=draft.text, structured=draft.structured,
                                source={"extractor": f"{table_backend}.find_tables"},
                            ))
                            chunk_index += 1

            flush_narrative()
        finally:
            if pdfplumber_doc:
                pdfplumber_doc.close()

        glossary_entries = extract_all_entries(all_paragraphs, doc_name)

        link_neighbors(all_chunks)
        stats = {
            "status": "ok",
            "n_pages": doc.page_count,
            "n_chunks_narrative": sum(1 for c in all_chunks if c.chunk_type == "narrative"),
            "n_chunks_table": sum(1 for c in all_chunks if c.chunk_type == "table"),
            "n_glossary_entries": len(glossary_entries),
        }
        return all_chunks, glossary_entries, stats
    finally:
        doc.close()


def _process_one(args: tuple) -> tuple[str, dict]:
    pdf_path, catalog_entry, config, table_backend, output_dir, overwrite = args
    doc_name = catalog_entry["doc_name"]
    chunks_path = output_dir / "chunks" / f"{doc_name}.jsonl"
    if chunks_path.exists() and not overwrite:
        return doc_name, {"status": "skipped"}

    start = time.monotonic()
    try:
        chunks, glossary_entries, stats = process_document(pdf_path, catalog_entry, config, table_backend, output_dir)
    except Exception as exc:  # noqa: BLE001 - isolate one bad doc from the whole batch
        return doc_name, {"status": "failed", "error": f"{exc}\n{traceback.format_exc()}"}

    stats["timing_s"] = round(time.monotonic() - start, 2)
    if stats["status"] != "ok":
        return doc_name, stats

    write_jsonl(chunks_path, [c.to_json() for c in chunks])
    write_jsonl(output_dir / "glossary" / f"{doc_name}.glossary.jsonl", [e.to_json() for e in glossary_entries])
    return doc_name, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text/tables/glossary chunks from downloaded FinanceBench PDFs.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to financebench_document_information.jsonl")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Directory containing downloaded PDFs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write processed chunks/glossary into")
    parser.add_argument("--chunk-size", type=int, default=450, help="Target narrative chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=60, help="Narrative chunk overlap in tokens")
    parser.add_argument("--table-backend", choices=["pymupdf", "pdfplumber"], default="pymupdf", help="Table extraction backend")
    parser.add_argument("--table-format", choices=["markdown", "html", "sentences"], default="markdown", help="Serialization format for table chunk text")
    parser.add_argument("--no-table-images", action="store_true", help="Skip cropping/saving a PNG/JPEG of each detected table region")
    parser.add_argument("--table-image-format", choices=["png", "jpg"], default="png", help="Image format for cropped table images")
    parser.add_argument("--table-image-dpi", type=int, default=300, help="Resolution (DPI) for cropped table images")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (CPU-bound parsing)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents (useful for testing)")
    parser.add_argument("--doc-name", type=str, default=None, help="Only process this single doc_name")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess docs that already have output chunks")
    parser.add_argument("--report-path", type=Path, default=None, help="Where to write the CSV status report (default: <output-dir>/extraction_report.csv)")
    args = parser.parse_args()

    config = ExtractionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        table_backend=args.table_backend,
        table_format=args.table_format,
        table_images_enabled=not args.no_table_images,
        table_image_format=args.table_image_format,
        table_image_dpi=args.table_image_dpi,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.write(args.output_dir / "extraction_config.json")
    report_path = args.report_path or (args.output_dir / "extraction_report.csv")

    catalog = load_catalog(args.catalog)
    if args.doc_name:
        catalog = {args.doc_name: catalog[args.doc_name]} if args.doc_name in catalog else {}

    jobs = []
    missing = []
    for doc_name, entry in sorted(catalog.items()):
        pdf_path = args.pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            missing.append(doc_name)
            continue
        jobs.append((pdf_path, entry, config, args.table_backend, args.output_dir, args.overwrite))

    if args.limit:
        jobs = jobs[: args.limit]

    results: dict[str, dict] = {name: {"status": "source_missing"} for name in missing}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting", unit="doc"):
            doc_name, stats = future.result()
            results[doc_name] = stats
            mark = {"ok": "OK", "skipped": "--", "failed": "XX", "source_missing": "??", "likely_scanned_or_image": "!!"}.get(stats["status"], "??")
            tqdm.write(f"[{mark}] {stats['status']:<24} {doc_name}")

    fieldnames = ["doc_name", "status", "n_pages", "n_chunks_narrative", "n_chunks_table", "n_glossary_entries", "timing_s", "error"]
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc_name, stats in sorted(results.items()):
            writer.writerow({"doc_name": doc_name, **{k: stats.get(k, "") for k in fieldnames if k != "doc_name"}})

    # Corpus-level glossary rollup, built from whatever per-doc glossary files exist on disk.
    from extract.schema import GlossaryEntry as GE
    from extract.schema import read_jsonl
    all_doc_entries = []
    for path in sorted((args.output_dir / "glossary").glob("*.glossary.jsonl")):
        all_doc_entries.append([GE(**record) for record in read_jsonl(path)])
    if all_doc_entries:
        rollup = build_global_rollup(all_doc_entries)
        write_jsonl(args.output_dir / "glossary_global.jsonl", rollup)

    counts: dict[str, int] = {}
    for stats in results.values():
        counts[stats["status"]] = counts.get(stats["status"], 0) + 1
    print(f"\n{len(results)} total: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()
