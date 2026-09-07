"""Unit tests for composite boolean facet expression parser and bitmask pre-filtering."""

from __future__ import annotations

import numpy as np
import pytest

from xists.search.embed import EmbeddingConfig
from xists.search.facets import (
    AndNode,
    FacetParseError,
    NotNode,
    OrNode,
    PredicateNode,
    evaluate_ast_mask,
    evaluate_ast_record,
    parse_facet_query,
    parse_filter_criteria,
    tokenize_facet_query,
)
from xists.search.query import PreparedIndex, rank

CONFIG = EmbeddingConfig(api_key="k", base_url="http://localhost/v1", model="bge-m3")


def test_tokenize_facet_query_operators_and_terms() -> None:
    tokens = tokenize_facet_query("(lang:python OR lang:rust) AND stars:>1000 AND NOT is:archived")
    kinds = [t.kind for t in tokens]
    values = [t.value for t in tokens]

    assert kinds == [
        "LPAREN",
        "TERM",
        "OR",
        "TERM",
        "RPAREN",
        "AND",
        "TERM",
        "AND",
        "NOT",
        "TERM",
    ]
    assert values == [
        "(",
        "lang:python",
        "OR",
        "lang:rust",
        ")",
        "AND",
        "stars:>1000",
        "AND",
        "NOT",
        "is:archived",
    ]


def test_tokenize_facet_query_symbols_and_quotes() -> None:
    tokens = tokenize_facet_query(
        'lang:"c++" && (eco:npm || eco:cargo) !is:archived stars:1k..50k -is:disabled'
    )
    kinds = [t.kind for t in tokens]
    values = [t.value for t in tokens]

    assert 'lang:"c++"' in values
    assert "AND" in kinds  # from &&
    assert "OR" in kinds  # from ||
    assert "NOT" in kinds  # from ! and -
    assert "stars:1k..50k" in values


def test_parse_simple_predicates() -> None:
    # 1. Language
    node_lang = parse_facet_query("lang:python")
    assert isinstance(node_lang, PredicateNode)
    assert node_lang.field == "language"
    assert node_lang.value == "python"

    # 2. Stars with comparison operators and suffixes
    node_stars_gt = parse_facet_query("stars:>10k")
    assert isinstance(node_stars_gt, PredicateNode)
    assert node_stars_gt.field == "stars"
    assert node_stars_gt.operator == "gt"
    assert node_stars_gt.value == 10000

    node_stars_range = parse_facet_query("stars:500..2000")
    assert isinstance(node_stars_range, PredicateNode)
    assert node_stars_range.field == "stars"
    assert node_stars_range.operator == "range"
    assert node_stars_range.value == (500, 2000)

    # 3. Flags
    node_archived = parse_facet_query("is:archived")
    assert isinstance(node_archived, PredicateNode)
    assert node_archived.field == "archived"
    assert node_archived.value is True

    node_not_archived = parse_facet_query("not:archived")
    assert isinstance(node_not_archived, PredicateNode)
    assert node_not_archived.field == "archived"
    assert node_not_archived.value is False


def test_parse_boolean_precedence_and_grouping() -> None:
    # NOT binds tighter than AND; AND binds tighter than OR
    # "lang:python OR lang:rust AND NOT is:archived" -> OR(lang:python, AND(lang:rust, NOT(is:archived)))
    ast = parse_facet_query("lang:python OR lang:rust AND NOT is:archived")
    assert isinstance(ast, OrNode)
    assert len(ast.children) == 2
    assert isinstance(ast.children[0], PredicateNode)
    assert ast.children[0].field == "language"

    right = ast.children[1]
    assert isinstance(right, AndNode)
    assert len(right.children) == 2
    assert isinstance(right.children[0], PredicateNode)
    assert isinstance(right.children[1], NotNode)

    # Parentheses override precedence
    # "(lang:python OR lang:rust) AND stars:>1000"
    ast_grouped = parse_facet_query("(lang:python OR lang:rust) AND stars:>1000")
    assert isinstance(ast_grouped, AndNode)
    assert len(ast_grouped.children) == 2
    assert isinstance(ast_grouped.children[0], OrNode)
    assert isinstance(ast_grouped.children[1], PredicateNode)


def test_parse_implicit_and_juxtaposition() -> None:
    # Juxtaposition without explicit AND keyword
    ast = parse_facet_query("lang:python stars:>500 eco:pypi !is:archived")
    assert isinstance(ast, AndNode)
    assert len(ast.children) == 4
    assert ast.children[0].field == "language"  # type: ignore[attr-defined]
    assert ast.children[1].field == "stars"  # type: ignore[attr-defined]
    assert ast.children[2].field == "ecosystem"  # type: ignore[attr-defined]
    assert isinstance(ast.children[3], NotNode)


def test_parse_syntax_errors() -> None:
    with pytest.raises(FacetParseError, match="Unclosed parenthesis"):
        parse_facet_query("(lang:python OR lang:rust")

    with pytest.raises(FacetParseError, match="Unexpected token"):
        parse_facet_query("lang:python)")


def test_parse_filter_criteria_normalization() -> None:
    # String input
    ast_str = parse_filter_criteria("lang:go stars:>100")
    assert isinstance(ast_str, AndNode)

    # Dict input with structured keys
    ast_dict = parse_filter_criteria(
        {
            "language": "rust",
            "min_stars": 1000,
            "max_stars": 50000,
            "ecosystem": ["cargo", "crates.io"],
            "include_archived": False,
        }
    )
    assert isinstance(ast_dict, AndNode)

    # None input
    assert parse_filter_criteria(None) is None
    assert parse_filter_criteria({}) is None


def test_evaluate_ast_record() -> None:
    record = {
        "repo_id": "astral-sh/uv",
        "name": "uv",
        "github": {
            "language": "Rust",
            "stars": 45000,
            "forks": 1200,
            "license": "MIT",
            "archived": False,
            "topics": ["python", "package-manager", "cli"],
        },
        "llm_profile": {
            "project_type": "cli_tool",
            "ecosystem": ["cargo", "pypi"],
        },
    }

    # Match composite boolean
    ast_match = parse_facet_query(
        "(lang:rust OR lang:python) AND stars:>10k AND type:cli_tool AND NOT is:archived"
    )
    assert ast_match is not None
    assert evaluate_ast_record(ast_match, record) is True

    # Mismatch language
    ast_mismatch_lang = parse_facet_query("lang:go")
    assert ast_mismatch_lang is not None
    assert evaluate_ast_record(ast_mismatch_lang, record) is False

    # Mismatch stars range
    ast_mismatch_stars = parse_facet_query("stars:100..5000")
    assert ast_mismatch_stars is not None
    assert evaluate_ast_record(ast_mismatch_stars, record) is False


def test_evaluate_ast_mask_on_prepared_index() -> None:
    entries = [
        {
            "repo_id": "fastapi/fastapi",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "fastapi",
                "summary": "FastAPI web framework",
                "language": "Python",
                "stars": 75000,
                "forks": 6000,
                "license": "MIT",
                "ecosystem": ["pypi"],
                "project_type": "framework",
                "topics": ["async", "api", "web"],
                "archived": False,
            },
        },
        {
            "repo_id": "astral-sh/uv",
            "vector": [0.8, 0.6],
            "metadata": {
                "name": "uv",
                "summary": "Extremely fast Python package manager in Rust",
                "language": "Rust",
                "stars": 42000,
                "forks": 1100,
                "license": "Apache-2.0",
                "ecosystem": ["cargo", "pypi"],
                "project_type": "cli_tool",
                "topics": ["packaging", "installer", "cli"],
                "archived": False,
            },
        },
        {
            "repo_id": "expressjs/express",
            "vector": [0.6, 0.8],
            "metadata": {
                "name": "express",
                "summary": "Fast Node.js web framework",
                "language": "JavaScript",
                "stars": 64000,
                "forks": 12000,
                "license": "MIT",
                "ecosystem": ["npm"],
                "project_type": "framework",
                "topics": ["nodejs", "web", "server"],
                "archived": False,
            },
        },
        {
            "repo_id": "old/legacy-project",
            "vector": [0.0, 1.0],
            "metadata": {
                "name": "legacy-project",
                "summary": "Archived Python tool",
                "language": "Python",
                "stars": 1200,
                "forks": 80,
                "license": "GPL-3.0",
                "ecosystem": ["pypi"],
                "project_type": "tool",
                "topics": ["legacy"],
                "archived": True,
            },
        },
    ]

    index_dict = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "record_count": 4,
        "vectors_normalized": True,
        "_matrix": np.array([e["vector"] for e in entries], dtype=np.float32),
        "vectors": entries,
    }

    prepared = PreparedIndex.from_dict(index_dict, CONFIG)

    # 1. Complex composite expression: (lang:python OR lang:rust) AND stars:>20k AND NOT is:archived
    ast = parse_facet_query("(lang:python OR lang:rust) AND stars:>20k AND NOT is:archived")
    assert ast is not None
    mask = evaluate_ast_mask(ast, prepared)
    assert np.array_equal(mask, np.array([True, True, False, False], dtype=bool))
    assert np.array_equal(
        prepared.compute_filter_mask(
            "(lang:python OR lang:rust) AND stars:>20k AND NOT is:archived"
        ),
        mask,
    )

    # 2. Ecosystem OR expression: eco:npm OR eco:cargo
    mask_eco = prepared.compute_filter_mask("eco:npm OR eco:cargo")
    assert mask_eco is not None
    assert np.array_equal(mask_eco, np.array([False, True, True, False], dtype=bool))

    # 3. Topic filtering with NOT archived: topic:web AND !is:archived
    mask_web = prepared.compute_filter_mask("topic:web AND !is:archived")
    assert mask_web is not None
    assert np.array_equal(mask_web, np.array([True, False, True, False], dtype=bool))

    # 4. Forks comparison: forks:>5000
    mask_forks = prepared.compute_filter_mask("forks:>5000")
    assert mask_forks is not None
    assert np.array_equal(mask_forks, np.array([True, False, True, False], dtype=bool))

    # 5. License and stars range: license:mit AND stars:60k..80k
    mask_lic = prepared.compute_filter_mask("license:mit AND stars:60k..80k")
    assert mask_lic is not None
    assert np.array_equal(mask_lic, np.array([True, False, True, False], dtype=bool))


def test_rank_with_composite_boolean_facet_expression() -> None:
    entries = [
        {
            "repo_id": "fastapi/fastapi",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "fastapi",
                "summary": "FastAPI framework for Python",
                "language": "Python",
                "stars": 75000,
                "license": "MIT",
            },
        },
        {
            "repo_id": "actix/actix-web",
            "vector": [0.95, 0.312],
            "metadata": {
                "name": "actix-web",
                "summary": "Actix Web is a powerful Rust web framework",
                "language": "Rust",
                "stars": 21000,
                "license": "MIT",
            },
        },
        {
            "repo_id": "expressjs/express",
            "vector": [0.9, 0.4358],
            "metadata": {
                "name": "express",
                "summary": "Express web framework for Node.js",
                "language": "JavaScript",
                "stars": 64000,
                "license": "MIT",
            },
        },
    ]

    index = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "record_count": 3,
        "vectors_normalized": True,
        "_matrix": np.array([e["vector"] for e in entries], dtype=np.float32),
        "vectors": entries,
    }

    query_vec = [1.0, 0.0]

    # Search with string expression: "(lang:python OR lang:rust) AND stars:>25k"
    # Should match only fastapi (stars 75k > 25k, python) and exclude actix-web (21k <= 25k) and express (js)
    res = rank(
        "web framework",
        index,
        CONFIG,
        ranking_strategy="semantic",
        filters="(lang:python OR lang:rust) AND stars:>25k",
        embed=lambda c, q: query_vec,
    )
    assert not res["abstained"]
    assert len(res["results"]) == 1
    assert res["results"][0]["repo_id"] == "fastapi/fastapi"
    assert res["considered"] == 1

    # Search with OR expression allowing Rust: "(lang:python OR lang:rust) AND stars:>20k"
    res_both = rank(
        "web framework",
        index,
        CONFIG,
        ranking_strategy="semantic",
        filters="(lang:python OR lang:rust) AND stars:>20k",
        embed=lambda c, q: query_vec,
    )
    assert len(res_both["results"]) == 2
    rids = {r["repo_id"] for r in res_both["results"]}
    assert rids == {"fastapi/fastapi", "actix/actix-web"}
    assert res_both["considered"] == 2


def test_rank_with_hybrid_and_boolean_filter() -> None:
    entries = [
        {
            "repo_id": "fastapi/fastapi",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "fastapi",
                "summary": "FastAPI framework for Python",
                "language": "Python",
                "stars": 75000,
                "license": "MIT",
                "topics": ["async", "web"],
            },
        },
        {
            "repo_id": "actix/actix-web",
            "vector": [0.9, 0.4358],
            "metadata": {
                "name": "actix-web",
                "summary": "Actix web framework in Rust",
                "language": "Rust",
                "stars": 21000,
                "license": "MIT",
                "topics": ["async", "web"],
            },
        },
        {
            "repo_id": "django/django",
            "vector": [0.7, 0.714],
            "metadata": {
                "name": "django",
                "summary": "The web framework for perfectionists with deadlines",
                "language": "Python",
                "stars": 80000,
                "license": "BSD-3-Clause",
                "topics": ["web", "framework"],
            },
        },
    ]

    index = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "record_count": 3,
        "vectors_normalized": True,
        "_matrix": np.array([e["vector"] for e in entries], dtype=np.float32),
        "vectors": entries,
    }

    # Filter for MIT license and async topic: "license:mit AND topics:async"
    res = rank(
        "async framework",
        index,
        CONFIG,
        ranking_strategy="hybrid",
        filters="license:mit AND topics:async",
        embed=lambda c, q: [1.0, 0.0],
    )
    assert not res["abstained"]
    assert len(res["results"]) == 2
    rids = [r["repo_id"] for r in res["results"]]
    assert "django/django" not in rids
    assert "fastapi/fastapi" in rids


def test_similar_with_boolean_filter() -> None:
    from xists.search.similar import find_similar_prepared

    entries = [
        {
            "repo_id": "fastapi/fastapi",
            "vector": [1.0, 0.0],
            "metadata": {
                "name": "fastapi",
                "language": "Python",
                "stars": 75000,
                "ecosystem": ["pypi"],
            },
        },
        {
            "repo_id": "pallets/flask",
            "vector": [0.99, 0.1],
            "metadata": {
                "name": "flask",
                "language": "Python",
                "stars": 68000,
                "ecosystem": ["pypi"],
            },
        },
        {
            "repo_id": "actix/actix-web",
            "vector": [0.95, 0.3],
            "metadata": {
                "name": "actix-web",
                "language": "Rust",
                "stars": 21000,
                "ecosystem": ["cargo"],
            },
        },
    ]

    index = {
        "index_version": 4,
        "embedding_input_version": 3,
        "record_schema_version": 2,
        "embedding_model": "bge-m3",
        "dimension": 2,
        "record_count": 3,
        "vectors_normalized": True,
        "_matrix": np.array([e["vector"] for e in entries], dtype=np.float32),
        "vectors": entries,
    }

    prepared = PreparedIndex.from_dict(index, CONFIG)

    # Filter similar projects by Rust only
    sim_rust = find_similar_prepared("fastapi/fastapi", prepared, filters="lang:rust")
    assert len(sim_rust["results"]) == 1
    assert sim_rust["results"][0]["repo_id"] == "actix/actix-web"

    # Filter similar projects by Python only
    sim_py = find_similar_prepared("fastapi/fastapi", prepared, filters="lang:python")
    assert len(sim_py["results"]) == 1
    assert sim_py["results"][0]["repo_id"] == "pallets/flask"


def test_starter_metadata_search_with_boolean_filter() -> None:
    from xists.starter import starter_metadata_search

    records = [
        {
            "repo_id": "astral-sh/uv",
            "name": "uv",
            "github": {"language": "Rust", "stars": 45000, "archived": False},
            "llm_profile": {
                "summary": "Extremely fast code formatter and installer in Rust",
                "project_type": "cli_tool",
                "ecosystem": ["cargo", "pypi"],
            },
        },
        {
            "repo_id": "psf/black",
            "name": "black",
            "github": {"language": "Python", "stars": 38000, "archived": False},
            "llm_profile": {
                "summary": "The uncompromising Python code formatter",
                "project_type": "cli_tool",
                "ecosystem": ["pypi"],
            },
        },
    ]

    res = starter_metadata_search("formatter", records=records, filters="lang:rust")
    assert len(res["results"]) == 1
    assert res["results"][0]["repo_id"] == "astral-sh/uv"
