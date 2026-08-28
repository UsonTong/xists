"""Build bundled starter demo dataset (200 top repos) for xists."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from xists.records import RECORD_SCHEMA_VERSION, records_validation_report
from xists.search.embed import EMBEDDING_INPUT_VERSION, embedding_input_fingerprint
from xists.search.index import INDEX_VERSION, decode_vector, entry_metadata, save_index


def build_starter_200(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.json"
    index_path = output_dir / "index.json"

    # Load master records and index
    with open("records.json", encoding="utf-8") as f:
        all_records = json.load(f)

    with open("index.json", encoding="utf-8") as f:
        master_index = json.load(f)

    master_vectors_by_id = {
        v["repo_id"].lower(): v for v in master_index.get("vectors", []) if "repo_id" in v
    }
    records_by_id = {r["repo_id"].lower(): r for r in all_records if "repo_id" in r}

    # Select 200 high-profile repos: Pinned Landmarks + repos.txt + top starred
    pinned_ids = [
        "fastapi/fastapi",
        "flask/flask",
        "django/django",
        "expressjs/express",
        "facebook/react",
        "vuejs/vue",
        "vuejs/core",
        "sveltejs/svelte",
        "vllm-project/vllm",
        "ollama/ollama",
        "ggerganov/llama.cpp",
        "supabase/supabase",
        "pocketbase/pocketbase",
        "withastro/astro",
        "gohugoio/hugo",
        "astral-sh/uv",
        "astral-sh/ruff",
        "n8n-io/n8n",
        "qdrant/qdrant",
        "microsoft/vscode",
    ]

    repos_txt_lines = [
        line.strip().lower().replace("https://github.com/", "")
        for line in Path("repos.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    selected_records: list[dict] = []
    seen_ids = set()

    # 1. Pinned landmark repos
    for pid in pinned_ids:
        pid_low = pid.lower()
        if pid_low in records_by_id and pid_low in master_vectors_by_id and pid_low not in seen_ids:
            selected_records.append(records_by_id[pid_low])
            seen_ids.add(pid_low)

    # 2. From repos.txt
    for rid in repos_txt_lines:
        if rid in records_by_id and rid in master_vectors_by_id and rid not in seen_ids:
            selected_records.append(records_by_id[rid])
            seen_ids.add(rid)

    # 3. Fill to 200 from highest starred
    if len(selected_records) < 200:
        candidates = [
            r
            for r in all_records
            if r["repo_id"].lower() in master_vectors_by_id and r["repo_id"].lower() not in seen_ids
        ]
        candidates.sort(key=lambda r: int(r.get("github", {}).get("stars") or 0), reverse=True)
        for c in candidates:
            selected_records.append(c)
            seen_ids.add(c["repo_id"].lower())
            if len(selected_records) == 200:
                break

    # Strip large redundant readmes to keep package size under 1.5MB
    compact_records = []
    for r in selected_records:
        r_copy = dict(r)
        readme = r_copy.get("readme") or ""
        if len(readme) > 400:
            r_copy["readme"] = readme[:400] + "\n..."
        compact_records.append(r_copy)

    # Validate records schema
    val = records_validation_report(
        compact_records,
        expected_schema_version=RECORD_SCHEMA_VERSION,
        expected_profile_prompt_version=2,
    )
    assert not val["errors"], f"Validation failed: {val['errors']}"

    records_path.write_text(json.dumps(compact_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(compact_records)} starter records -> {records_path}")

    # Build vector matrix and vectors metadata
    dimension = master_index.get("dimension", 2048)
    raw_vectors = []
    vectors_meta = []

    for r in compact_records:
        rid = r["repo_id"].lower()
        v_entry = master_vectors_by_id[rid]
        vec = decode_vector(v_entry.get("vector"))
        assert vec is not None and len(vec) == dimension, f"Invalid vector for {r['repo_id']}"
        raw_vectors.append(vec)
        vectors_meta.append({
            "repo_id": r["repo_id"],
            "embedding_input_fingerprint": embedding_input_fingerprint(r),
            "metadata": entry_metadata(r),
        })

    matrix = np.asarray(raw_vectors, dtype=np.float32)

    index_doc = {
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": master_index.get("embedding_model", "xists-starter-v1"),
        "embedding_base_url": master_index.get("embedding_base_url"),
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": dimension,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(compact_records),
        "skipped": [],
        "vectors": vectors_meta,
    }

    save_index(index_path, index_doc, matrix=matrix, version=4)
    print(f"Generated starter index -> {index_path} and {index_path.stem}.vectors.npy")


if __name__ == "__main__":
    build_starter_200(Path("src/xists/starter"))
