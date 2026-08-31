"""Unit tests for in-memory BM25 sparse index and tokenization."""

from __future__ import annotations

import numpy as np

from xists.search.bm25 import BM25Index, bm25_text_from_entry, tokenize


def test_tokenize_ascii_and_technical_symbols() -> None:
    tokens = tokenize("Fast, lightweight C++ and Rust parser for Next.js v14.0 #tag")
    assert "fast" in tokens
    assert "lightweight" in tokens
    assert "c++" in tokens
    assert "rust" in tokens
    assert "parser" in tokens
    assert "next.js" in tokens
    assert "v14.0" in tokens
    assert "#tag" in tokens or "tag" in tokens


def test_tokenize_cjk_ngrams() -> None:
    tokens = tokenize("自托管大模型应用")
    assert "自托管" in tokens
    assert "托管" in tokens
    assert "大模型" in tokens
    assert "模型" in tokens
    assert "应用" in tokens


def test_bm25_text_from_entry() -> None:
    entry = {
        "repo_id": "astral-sh/uv",
        "metadata": {
            "name": "uv",
            "description": "An extremely fast Python package and project manager.",
            "language": "Rust",
            "aliases": ["uv-cli"],
            "topics": ["python", "packaging", "installer"],
            "replaces": ["pip", "poetry"],
        },
    }
    text = bm25_text_from_entry(entry)
    assert "astral-sh/uv" in text
    assert "uv" in text
    assert "astral-sh" in text
    assert "extremely fast Python package" in text
    assert "uv-cli" in text
    assert "packaging" in text
    assert "poetry" in text


def test_bm25_empty_index() -> None:
    index = BM25Index.build_from_entries([])
    assert index.doc_count == 0
    scores = index.score_query("fast rust web framework")
    assert len(scores) == 0


def test_bm25_single_doc_and_exact_keyword_recall() -> None:
    entries = [
        {
            "repo_id": "astral-sh/uv",
            "metadata": {
                "name": "uv",
                "summary": "An extremely fast Python package installer and resolver written in Rust.",
                "aliases": ["uv"],
                "topics": ["python", "packaging", "package-manager"],
            },
        },
        {
            "repo_id": "astral-sh/ruff",
            "metadata": {
                "name": "ruff",
                "summary": "An extremely fast Python linter and code formatter written in Rust.",
                "aliases": ["ruff"],
                "topics": ["python", "linter", "formatter"],
            },
        },
        {
            "repo_id": "expressjs/express",
            "metadata": {
                "name": "express",
                "summary": "Fast, unopinionated, minimalist web framework for Node.js.",
                "aliases": ["express"],
                "topics": ["nodejs", "web", "framework"],
            },
        },
    ]

    index = BM25Index.build_from_entries(entries)
    assert index.doc_count == 3

    # Query for exact keyword 'uv'
    scores_uv = index.score_query("uv")
    assert scores_uv[0] > 0
    assert scores_uv[1] == 0
    assert scores_uv[2] == 0
    assert np.argmax(scores_uv) == 0

    # Query for 'linter formatter'
    scores_lint = index.score_query("linter formatter")
    assert scores_lint[1] > scores_lint[0]
    assert np.argmax(scores_lint) == 1

    # Query for 'web framework'
    scores_web = index.score_query("web framework")
    assert scores_web[2] > scores_web[0]
    assert np.argmax(scores_web) == 2


def test_bm25_idf_rarity_weighting() -> None:
    # 'python' appears in 3 docs, 'parquet' appears in only 1 doc
    entries = [
        {"repo_id": "a/p1", "metadata": {"name": "p1", "summary": "python library for web"}},
        {"repo_id": "a/p2", "metadata": {"name": "p2", "summary": "python data analytics tool"}},
        {
            "repo_id": "a/p3",
            "metadata": {"name": "p3", "summary": "python fast parquet file reader"},
        },
    ]
    index = BM25Index.build_from_entries(entries)

    # Inverted index should have higher IDF for 'parquet' than 'python'
    assert index.postings["parquet"].idf > index.postings["python"].idf

    # Query 'python parquet' should strongly favor p3
    scores = index.score_query("python parquet")
    assert np.argmax(scores) == 2
