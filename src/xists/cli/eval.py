"""Evaluation CLI commands (eval run, eval inspect, eval cases)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from xists.cli.common import (
    _load_canonical_queries,
    _prepare_query_transforms,
    _print_embedding_error,
    write_json,
)
from xists.eval.inspect import inspect_report, load_report
from xists.eval.run import evaluate_dataset
from xists.eval.schema import EvaluationDatasetError, load_dataset
from xists.profile.llm import LLMError, LLMNotConfiguredError, llm_config_from_env
from xists.search.embed import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    embedding_config_from_env,
)
from xists.search.query import IndexMismatchError, _query_intent
from xists.search.rerank import (
    RerankerError,
    RerankerNotConfiguredError,
    rerank_documents,
    reranker_config_from_env,
)
from xists.search.transform import (
    QueryTransformError,
    QueryTransformNotConfiguredError,
    query_variants,
)


def eval_run(args: argparse.Namespace) -> int:
    try:
        config = embedding_config_from_env()
    except EmbeddingNotConfiguredError as error:
        print(str(error), file=sys.stderr)
        return 2

    llm_judge_config = None
    if args.llm_judge:
        try:
            llm_judge_config = llm_config_from_env()
        except LLMNotConfiguredError as error:
            print(str(error), file=sys.stderr)
            return 2
        if args.records is None:
            print("--records is required when --llm-judge is enabled", file=sys.stderr)
            return 2

    if not args.index.exists():
        print(
            f"Index file not found: {args.index}. Run 'xists index build' first.", file=sys.stderr
        )
        return 2

    rerank = None
    if args.ranking_strategy == "rerank":
        try:
            reranker_config = reranker_config_from_env()
        except RerankerNotConfiguredError as error:
            print(str(error), file=sys.stderr)
            return 2

        def rerank(query: str, documents: list[str]) -> list[float]:
            return rerank_documents(reranker_config, query, documents)

    try:
        transform_kwargs: dict[str, Any] = {}
        if args.canonical_queries is not None and args.query_transform_mode == "off":
            raise QueryTransformError(
                "--canonical-queries requires --query-transform-mode canonical or merge"
            )
        if args.query_transform_mode != "off":
            cases = load_dataset(args.cases)["cases"]
            queries = [case["query"] for case in cases]
            if args.canonical_queries is None:
                variants, rerank_queries, transform_model = _prepare_query_transforms(
                    queries, args.query_transform_mode
                )
            else:
                rerank_queries = _load_canonical_queries(args.canonical_queries, cases)
                variants = [
                    query_variants(query, canonical, args.query_transform_mode)
                    for query, canonical in zip(queries, rerank_queries)
                ]
                transform_model = "frozen-canonical-queries"
            transform_kwargs = {
                "query_variants": variants,
                "rerank_queries": rerank_queries,
                "query_transform_mode": args.query_transform_mode,
                "query_transform_model": transform_model,
            }
        _evaluate_dataset = getattr(
            sys.modules.get("xists.cli"), "evaluate_dataset", evaluate_dataset
        )
        report = _evaluate_dataset(
            args.cases,
            args.index,
            config,
            top_k=args.top_k,
            batch_size=args.batch_size,
            llm_judge_config=llm_judge_config,
            records_path=args.records,
            ranking_strategy=args.ranking_strategy,
            rerank=rerank,
            rerank_candidate_limit=args.rerank_candidates,
            exploratory_threshold=args.exploratory_threshold,
            rerank_abstain_threshold=args.rerank_abstain_threshold,
            confidence_calibration=args.confidence_calibration,
            **transform_kwargs,
        )
    except EmbeddingError as error:
        _print_embedding_error(error, command="eval run")
        return 1
    except (
        EvaluationDatasetError,
        FileNotFoundError,
        IndexMismatchError,
        QueryTransformError,
        QueryTransformNotConfiguredError,
        ValueError,
        LLMError,
        RerankerError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def eval_inspect(args: argparse.Namespace) -> int:
    try:
        report = load_report(args.report)
        payload = inspect_report(
            report,
            status=args.status,
            limit=args.limit,
            include_exact=args.include_exact,
            tag=args.tag,
            intent=args.query_intent,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def eval_cases(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.cases)
    except EvaluationDatasetError as error:
        print(str(error), file=sys.stderr)
        return 1

    selected: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        if args.tag and args.tag not in case.get("tags", []):
            continue
        intent = _query_intent(case["query"]).get("type")
        if args.query_intent and intent != args.query_intent:
            continue
        selected.append(
            {
                "id": case["id"],
                "query": case["query"],
                "query_intent": intent,
                "expected_repo_id": case["expected_repo_id"],
                "acceptable": case.get("acceptable") or [],
                "acceptable_repo_ids": case.get("acceptable_repo_ids") or [],
                "acceptable_families": case.get("acceptable_families") or [],
                "tags": case.get("tags") or [],
                "notes": case.get("notes"),
            }
        )

    tag_counts = Counter(tag for case in dataset["cases"] for tag in case.get("tags", []))
    intent_counts = Counter(_query_intent(case["query"]).get("type") for case in dataset["cases"])
    payload = {
        "dataset_name": dataset.get("dataset_name"),
        "schema_version": dataset.get("schema_version"),
        "case_count": len(dataset["cases"]),
        "family_count": len(dataset.get("families") or {}),
        "tag_counts": [
            {"tag": tag, "count": count} for tag, count in tag_counts.most_common(args.limit)
        ],
        "query_intent_counts": [
            {"query_intent": intent, "count": count}
            for intent, count in intent_counts.most_common()
        ],
        "filter": {"tag": args.tag, "query_intent": args.query_intent, "limit": args.limit},
        "matching_count": len(selected),
        "inspected_count": min(len(selected), args.limit),
        "cases": selected[: args.limit],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "eval_cases",
    "eval_inspect",
    "eval_run",
]
