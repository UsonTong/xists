"""Generate synthetic records and index fixtures for performance testing.

Offline by design: vectors are seeded random unit vectors and profiles are
templated text. The records pass `xists records validate` and the index
matches `build_index()` output field-for-field so `index stats` and
`index verify` can consume the fixtures. `built_at` is a fixed constant so
the same seed always produces byte-identical output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from xists.profile.llm import PROFILE_PROMPT_VERSION
from xists.records import RECORD_SCHEMA_VERSION
from xists.search.embed import EMBEDDING_INPUT_VERSION, embedding_input_fingerprint
from xists.search.index import INDEX_VERSION, entry_metadata

SYNTHETIC_MODEL = "synthetic-test-model"
SYNTHETIC_BASE_URL = "http://synthetic.invalid/v1"
SYNTHETIC_BUILT_AT = "2026-01-01T00:00:00+00:00"

PROJECT_TYPES = ["library", "tool", "framework", "runtime", "app", "service"]
LANGUAGES = ["Python", "Rust", "JavaScript", "Go", "TypeScript"]
ECOSYSTEMS = [
    ["python", "llm"],
    ["rust", "devtools"],
    ["javascript", "web"],
    ["go", "infra"],
    ["typescript", "web"],
]


def make_synthetic_record(i: int) -> dict[str, Any]:
    repo_id = f"synthetic/repo-{i:06d}"
    name = f"repo-{i:06d}"
    project_type = PROJECT_TYPES[i % len(PROJECT_TYPES)]
    language = LANGUAGES[i % len(LANGUAGES)]
    ecosystem = ECOSYSTEMS[i % len(ECOSYSTEMS)]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "repo_id": repo_id,
        "url": f"https://github.com/{repo_id}",
        "name": name,
        "source": "synthetic",
        "github": {
            "description": f"Synthetic {project_type} number {i} for {ecosystem[0]} workloads.",
            "topics": [ecosystem[0], project_type],
            "language": language,
            "stars": (i * 37) % 5000,
            "forks": (i * 11) % 800,
            "archived": False,
            "disabled": False,
            "pushed_at": SYNTHETIC_BUILT_AT,
        },
        "readme": f"# {name}\n\nSynthetic fixture readme for performance testing.",
        "llm_profile": {
            "summary": f"Synthetic {project_type} project {i} used as a performance fixture.",
            "use_cases": [f"benchmarking {ecosystem[0]} {project_type} search"],
            "capabilities": [f"synthetic {project_type} capability {i}"],
            "not_for": ["production use"],
            "aliases": [name],
            "project_type": project_type,
            "ecosystem": ecosystem,
            "replaces": [],
            "related_projects": [],
            "search_text": (
                f"synthetic {project_type} for {' '.join(ecosystem)} workloads, "
                f"{language} performance fixture {name}, semantic search benchmark data"
            ),
            "search_phrases": [f"{ecosystem[0]} {project_type} fixture"],
            "confidence": "high",
            "abstained": False,
            "provider": "synthetic",
            "model": SYNTHETIC_MODEL,
            "generated_at": SYNTHETIC_BUILT_AT,
            "prompt_version": PROFILE_PROMPT_VERSION,
        },
    }


def generate_records(count: int) -> list[dict[str, Any]]:
    return [make_synthetic_record(i) for i in range(count)]


def generate_vectors(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    matrix = rng.standard_normal((count, dimension))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def generate_index(records: list[dict[str, Any]], dimension: int, seed: int) -> dict[str, Any]:
    matrix = generate_vectors(len(records), dimension, seed)
    vectors = [
        {
            "repo_id": record["repo_id"],
            "embedding_input_fingerprint": embedding_input_fingerprint(record),
            "metadata": entry_metadata(record),
            "vector": [round(float(value), 6) for value in matrix[i]],
        }
        for i, record in enumerate(records)
    ]
    return {
        "index_version": INDEX_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "embedding_model": SYNTHETIC_MODEL,
        "embedding_base_url": SYNTHETIC_BASE_URL,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "dimension": dimension,
        "built_at": SYNTHETIC_BUILT_AT,
        "record_count": len(vectors),
        "skipped": [],
        "vectors": vectors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True, help="Number of synthetic records to generate")
    parser.add_argument("--dimension", type=int, default=1024, help="Embedding vector dimension")
    parser.add_argument("--output-records", type=Path, required=True, help="Path for the records JSON")
    parser.add_argument("--output-index", type=Path, required=True, help="Path for the index JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for vector generation")
    args = parser.parse_args()

    records = generate_records(args.count)
    index = generate_index(records, args.dimension, args.seed)

    args.output_records.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    args.output_index.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.output_records}")
    print(f"wrote index ({args.count} vectors, dimension {args.dimension}) to {args.output_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
