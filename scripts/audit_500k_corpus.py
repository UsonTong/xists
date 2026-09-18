#!/usr/bin/env python3
"""Streaming Quality Audit for 500k Repository Corpus in xists.

This script performs a zero-memory-leak, high-performance streaming audit over
all 498,054 records in data/index_500k.meta.db without loading entire JSON
payloads into memory at once.
"""

from __future__ import annotations

import json
import resource
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Hard memory safety limit: 2.0 GB virtual memory guard
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except (ValueError, OSError):
    pass

DB_PATH = Path("data/index_500k.meta.db")
OUTPUT_REPORT_PATH = Path("data/audit_report_500k.json")
CHUNK_SIZE = 10000

SUSPICIOUS_NAMES = {"none", "null", "undefined", "test", "demo"}
REFUSAL_PHRASES = (
    "as an ai language model",
    "as a large language model",
    "i cannot assist with",
    "i cannot fulfill this request",
    "i am unable to provide",
    "sorry, i cannot",
    "i am an ai, i cannot",
    "as an ai, i cannot",
)


def run_audit() -> dict[str, Any]:
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting 500k corpus quality audit on {DB_PATH}...")
    start_time = time.monotonic()

    conn = sqlite3.connect(f"file:{DB_PATH.resolve().as_posix()}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM records")
    total_records = cursor.fetchone()[0]
    print(f"Total records in database: {total_records:,}")

    # Metrics
    processed_count = 0
    corrupted_json = 0
    invalid_repo_ids = 0
    missing_name = 0
    suspicious_name = 0
    missing_description = 0
    missing_summary = 0
    refusal_summary = 0
    missing_search_text = 0
    short_search_text = 0  # < 20 chars
    empty_capabilities = 0
    empty_use_cases = 0
    missing_language = 0
    missing_license = 0
    archived_count = 0
    disabled_count = 0
    negative_stars_forks = 0
    profile_abstained = 0
    low_confidence_count = 0  # < 0.3

    languages_counter: Counter[str] = Counter()
    confidence_tiers: Counter[str] = Counter()
    stars_tiers: Counter[str] = Counter()

    sample_defects: list[dict[str, Any]] = []

    completely_empty_content = 0

    cursor.execute(
        "SELECT doc_id, repo_id, name, language, stars, forks, archived, disabled, entry_json FROM records ORDER BY doc_id ASC"
    )

    while True:
        rows = cursor.fetchmany(CHUNK_SIZE)
        if not rows:
            break

        for doc_id, repo_id, db_name, db_lang, stars, forks, archived, disabled, entry_json in rows:
            processed_count += 1

            # 1. Parse JSON
            try:
                entry = json.loads(entry_json)
            except Exception as e:
                corrupted_json += 1
                if len(sample_defects) < 20:
                    sample_defects.append(
                        {"doc_id": doc_id, "repo_id": repo_id, "issue": f"json_decode_error: {e}"}
                    )
                continue

            metadata = entry.get("metadata") or {}

            # 2. repo_id check
            if (
                not repo_id
                or "/" not in repo_id
                or repo_id.startswith("/")
                or repo_id.endswith("/")
            ):
                invalid_repo_ids += 1
                if len(sample_defects) < 20:
                    sample_defects.append(
                        {"doc_id": doc_id, "repo_id": repo_id, "issue": "invalid_repo_id_format"}
                    )

            # 3. Name check
            name = metadata.get("name") or db_name or ""
            if not name.strip():
                missing_name += 1
            elif name.lower().strip() in SUSPICIOUS_NAMES:
                suspicious_name += 1

            # 4. Description check
            desc = (metadata.get("description") or "").strip()
            if not desc:
                missing_description += 1

            # 5. Summary & Profile check
            summary = (metadata.get("summary") or "").strip()
            if not summary:
                missing_summary += 1
            else:
                s_lower = summary.lower()
                if any(refusal in s_lower for refusal in REFUSAL_PHRASES):
                    refusal_summary += 1
                    if len(sample_defects) < 20:
                        sample_defects.append(
                            {
                                "doc_id": doc_id,
                                "repo_id": repo_id,
                                "issue": "refusal_phrase_in_summary",
                            }
                        )

            topics = metadata.get("topics") or []
            if not desc and not summary and not topics:
                completely_empty_content += 1

            # 6. search_text check
            stext = (metadata.get("search_text") or "").strip()
            if not stext:
                missing_search_text += 1
            elif len(stext) < 20:
                short_search_text += 1

            # 7. Capabilities & use cases
            caps = metadata.get("capabilities") or []
            if not caps:
                empty_capabilities += 1
            use_cases = metadata.get("use_cases") or []
            if not use_cases:
                empty_use_cases += 1

            # 8. Language & license
            lang = metadata.get("language") or db_lang or ""
            if not lang.strip():
                missing_language += 1
            else:
                languages_counter[lang.strip()] += 1

            license_val = metadata.get("license") or ""
            if not license_val.strip():
                missing_license += 1

            # 9. Lifecycle & stats
            if archived or metadata.get("archived"):
                archived_count += 1
            if disabled or metadata.get("disabled"):
                disabled_count += 1

            s_val = int(stars or 0)
            f_val = int(forks or 0)
            if s_val < 0 or f_val < 0:
                negative_stars_forks += 1

            if s_val >= 10000:
                stars_tiers[">=10k"] += 1
            elif s_val >= 1000:
                stars_tiers["1k-10k"] += 1
            elif s_val >= 100:
                stars_tiers["100-1k"] += 1
            elif s_val >= 10:
                stars_tiers["10-100"] += 1
            else:
                stars_tiers["<10"] += 1

            # 10. Confidence
            conf = metadata.get("confidence")
            if conf is None:
                confidence_tiers["null"] += 1
            elif isinstance(conf, (int, float)):
                if conf < 0.3:
                    low_confidence_count += 1
                    confidence_tiers["<0.3"] += 1
                elif conf < 0.6:
                    confidence_tiers["0.3-0.6"] += 1
                elif conf < 0.8:
                    confidence_tiers["0.6-0.8"] += 1
                else:
                    confidence_tiers[">=0.8"] += 1

            if metadata.get("abstained") is True:
                profile_abstained += 1

        if processed_count % 50000 == 0 or processed_count == total_records:
            elapsed = time.monotonic() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            print(
                f"  Processed {processed_count:,}/{total_records:,} records... ({rate:,.0f} records/s, elapsed: {elapsed:.1f}s)"
            )

    conn.close()
    elapsed = time.monotonic() - start_time

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_path": str(DB_PATH),
        "total_records": total_records,
        "processed_records": processed_count,
        "elapsed_seconds": round(elapsed, 2),
        "records_per_second": round(processed_count / elapsed if elapsed > 0 else 0, 1),
        "integrity": {
            "corrupted_json": corrupted_json,
            "invalid_repo_ids": invalid_repo_ids,
            "missing_name": missing_name,
            "suspicious_name": suspicious_name,
            "negative_stars_forks": negative_stars_forks,
        },
        "content_quality": {
            "missing_description": missing_description,
            "missing_summary": missing_summary,
            "refusal_summary": refusal_summary,
            "completely_empty_content": completely_empty_content,
            "missing_search_text": missing_search_text,
            "short_search_text": short_search_text,
            "empty_capabilities": empty_capabilities,
            "empty_use_cases": empty_use_cases,
            "missing_language": missing_language,
            "missing_license": missing_license,
            "profile_abstained": profile_abstained,
            "low_confidence": low_confidence_count,
        },
        "lifecycle": {
            "archived": archived_count,
            "disabled": disabled_count,
        },
        "distributions": {
            "stars": dict(stars_tiers),
            "confidence": dict(confidence_tiers),
            "top_15_languages": dict(languages_counter.most_common(15)),
        },
        "sample_defects": sample_defects,
        "overall_status": "PASS" if corrupted_json == 0 and invalid_repo_ids == 0 else "FAIL",
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nAudit completed in {elapsed:.2f}s! Overall Status: {report['overall_status']}")
    print(f"Report saved to: {OUTPUT_REPORT_PATH}")
    return report


if __name__ == "__main__":
    report = run_audit()
    print("\n--- Summary Report ---")
    print(f"Total Audited Records: {report['total_records']:,}")
    print(f"Corrupted JSON:        {report['integrity']['corrupted_json']}")
    print(f"Invalid Repo IDs:      {report['integrity']['invalid_repo_ids']}")
    print(f"Missing search_text:   {report['content_quality']['missing_search_text']}")
    print(f"Short search_text:     {report['content_quality']['short_search_text']}")
    print(f"Missing Summary:       {report['content_quality']['missing_summary']}")
    print(f"Refusal Summaries:     {report['content_quality']['refusal_summary']}")
    print(f"Missing Language:      {report['content_quality']['missing_language']}")
    print(f"Archived Count:        {report['lifecycle']['archived']:,}")
    print(f"Disabled Count:        {report['lifecycle']['disabled']}")
    print("Star Distribution:    ", report["distributions"]["stars"])
    print("Top Languages:        ", list(report["distributions"]["top_15_languages"].keys())[:8])
