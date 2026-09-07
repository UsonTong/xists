"""Unit tests for in-memory BM25 sparse index and tokenization."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_bm25_to_dict_and_from_dict_serialization() -> None:
    entries = [
        {
            "repo_id": "astral-sh/ruff",
            "metadata": {"name": "ruff", "summary": "Extremely fast Python linter."},
        },
        {
            "repo_id": "astral-sh/uv",
            "metadata": {"name": "uv", "summary": "Extremely fast Python package installer."},
        },
    ]
    original = BM25Index.build_from_entries(entries)
    data = original.to_dict()
    restored = BM25Index.from_dict(data)

    assert restored.doc_count == original.doc_count
    assert restored.avgdl == pytest.approx(original.avgdl)
    assert np.allclose(restored.doc_lengths, original.doc_lengths)
    assert set(restored.postings.keys()) == set(original.postings.keys())

    # Scoring parity
    orig_scores = original.score_query("fast python linter")
    rest_scores = restored.score_query("fast python linter")
    assert np.allclose(orig_scores, rest_scores)


def test_bm25_incremental_append() -> None:
    initial_entries = [
        {
            "repo_id": "astral-sh/ruff",
            "metadata": {"name": "ruff", "summary": "Extremely fast Python linter."},
        }
    ]
    new_entries = [
        {
            "repo_id": "astral-sh/uv",
            "metadata": {"name": "uv", "summary": "Extremely fast Python package installer."},
        },
        {
            "repo_id": "expressjs/express",
            "metadata": {"name": "express", "summary": "Fast web framework for Node.js."},
        },
    ]

    # 1. Full build of all 3
    full_entries = initial_entries + new_entries
    full_index = BM25Index.build_from_entries(full_entries)

    # 2. Incremental append
    incr_index = BM25Index.build_from_entries(initial_entries)
    incr_index.append_entries(new_entries)

    assert incr_index.doc_count == 3
    assert incr_index.doc_count == full_index.doc_count
    assert incr_index.avgdl == pytest.approx(full_index.avgdl)
    assert np.allclose(incr_index.doc_lengths, full_index.doc_lengths)

    # Check scores parity across all docs
    for query in ["ruff linter", "uv installer", "express nodejs web framework", "python"]:
        full_scores = full_index.score_query(query)
        incr_scores = incr_index.score_query(query)
        assert np.allclose(full_scores, incr_scores, atol=1e-5)


def test_bm25_prune_indices() -> None:
    entries = [
        {"repo_id": "repo/zero", "metadata": {"name": "zero", "summary": "First python tool."}},
        {"repo_id": "repo/one", "metadata": {"name": "one", "summary": "Second rust linter."}},
        {"repo_id": "repo/two", "metadata": {"name": "two", "summary": "Third web framework."}},
        {
            "repo_id": "repo/three",
            "metadata": {"name": "three", "summary": "Fourth python database."},
        },
    ]
    full_index = BM25Index.build_from_entries(entries)

    # Keep only index 0 and index 2
    pruned_index = full_index.prune_indices([0, 2])
    rebuilt_index = BM25Index.build_from_entries([entries[0], entries[2]])

    assert pruned_index.doc_count == 2
    assert pruned_index.doc_count == rebuilt_index.doc_count
    assert pruned_index.avgdl == pytest.approx(rebuilt_index.avgdl)
    assert np.allclose(pruned_index.doc_lengths, rebuilt_index.doc_lengths)

    for query in ["python tool", "web framework", "linter"]:
        p_scores = pruned_index.score_query(query)
        r_scores = rebuilt_index.score_query(query)
        assert np.allclose(p_scores, r_scores, atol=1e-5)
