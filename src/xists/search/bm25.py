"""In-memory Okapi BM25 sparse index for xists repository search."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

BM25_K1: float = 1.2
BM25_B: float = 0.75

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._#-]*")
CJK_RUN_RE = re.compile(r"[㐀-鿿]+")
CJK_TERM_LENGTHS = (3, 2)
CJK_TERM_LIMIT = 32


@lru_cache(maxsize=65536)
def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize ASCII and CJK text into normalized lowercase search tokens."""
    lowered = text.lower()
    ascii_tokens = TOKEN_RE.findall(lowered)
    tokens: list[str] = list(ascii_tokens)
    seen: set[str] = set(ascii_tokens)

    cjk_added = 0
    for run in CJK_RUN_RE.findall(lowered):
        for width in CJK_TERM_LENGTHS:
            if len(run) < width:
                continue
            for start in range(len(run) - width + 1):
                term = run[start : start + width]
                if term not in seen:
                    tokens.append(term)
                    seen.add(term)
                    cjk_added += 1
                    if cjk_added >= CJK_TERM_LIMIT:
                        return tuple(tokens)
    return tuple(tokens)


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def bm25_text_from_entry(entry: dict[str, Any]) -> str:
    """Extract search-relevant textual fields from an index entry for BM25 indexing."""
    raw_metadata = entry.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}

    parts: list[str] = []
    repo_id = str(entry.get("repo_id") or "").strip()
    if repo_id:
        parts.append(repo_id)
        if "/" in repo_id:
            parts.extend(repo_id.split("/"))

    for key in ("name", "description", "summary", "language", "project_type", "search_text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    for key in (
        "aliases",
        "topics",
        "use_cases",
        "capabilities",
        "ecosystem",
        "replaces",
        "related_projects",
        "search_phrases",
    ):
        parts.extend(_string_list(metadata.get(key)))

    return " ".join(parts)


@dataclass
class BM25Posting:
    """Sparse posting list for a single vocabulary term."""

    doc_indices: np.ndarray  # int32 array of document indices
    term_freqs: np.ndarray  # float32 array of raw term frequencies
    idf: float  # Robertson-Spärck Jones non-negative IDF


class BM25Index:
    """In-memory, vectorized Okapi BM25 sparse index using NumPy."""

    def __init__(
        self,
        *,
        doc_lengths: np.ndarray,
        avgdl: float,
        postings: dict[str, BM25Posting],
        doc_count: int,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.doc_lengths = doc_lengths
        self.avgdl = avgdl
        self.postings = postings
        self.doc_count = doc_count
        self.k1 = k1
        self.b = b

    def recompute_idfs(self) -> None:
        """Recalculate Robertson-Spärck Jones IDFs for all vocabulary terms based on current doc_count."""
        for posting in self.postings.values():
            n_q = len(posting.doc_indices)
            idf = float(math.log(1.0 + (self.doc_count - n_q + 0.5) / (n_q + 0.5)))
            posting.idf = idf

    def to_dict(self) -> dict[str, Any]:
        """Serialize BM25 index state to a JSON-compatible dictionary."""
        postings_data = {
            term: {
                "docs": posting.doc_indices.tolist(),
                "freqs": [round(float(f), 4) for f in posting.term_freqs],
                "idf": round(float(posting.idf), 6),
            }
            for term, posting in self.postings.items()
        }
        return {
            "doc_count": self.doc_count,
            "avgdl": round(float(self.avgdl), 4),
            "doc_lengths": [round(float(length), 4) for length in self.doc_lengths],
            "k1": self.k1,
            "b": self.b,
            "postings": postings_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Deserialize BM25 index from a JSON dictionary."""
        doc_count = int(data.get("doc_count", 0))
        doc_lengths = np.asarray(data.get("doc_lengths", []), dtype=np.float32)
        avgdl = float(data.get("avgdl", 0.0))
        k1 = float(data.get("k1", BM25_K1))
        b = float(data.get("b", BM25_B))

        postings: dict[str, BM25Posting] = {}
        for term, pdata in data.get("postings", {}).items():
            doc_ids = np.asarray(pdata.get("docs", []), dtype=np.int32)
            freqs = np.asarray(pdata.get("freqs", []), dtype=np.float32)
            idf = float(pdata.get("idf", 0.0))
            postings[term] = BM25Posting(doc_indices=doc_ids, term_freqs=freqs, idf=idf)

        return cls(
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            postings=postings,
            doc_count=doc_count,
            k1=k1,
            b=b,
        )

    def append_entries(self, new_entries: list[dict[str, Any]]) -> None:
        """Incrementally append new repository entries to the BM25 inverted index."""
        if not new_entries:
            return
        old_count = self.doc_count
        new_count = old_count + len(new_entries)
        new_doc_lengths: list[float] = []

        for sub_idx, entry in enumerate(new_entries):
            doc_idx = old_count + sub_idx
            doc_text = bm25_text_from_entry(entry)
            tokens = tokenize(doc_text)
            doc_length = float(len(tokens))
            new_doc_lengths.append(doc_length)

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self.postings:
                    posting = self.postings[term]
                    new_doc_indices = np.append(posting.doc_indices, np.int32(doc_idx))
                    new_term_freqs = np.append(posting.term_freqs, np.float32(count))
                    self.postings[term] = BM25Posting(
                        doc_indices=new_doc_indices,
                        term_freqs=new_term_freqs,
                        idf=posting.idf,
                    )
                else:
                    self.postings[term] = BM25Posting(
                        doc_indices=np.array([doc_idx], dtype=np.int32),
                        term_freqs=np.array([float(count)], dtype=np.float32),
                        idf=0.0,
                    )

        if len(self.doc_lengths) > 0:
            self.doc_lengths = np.concatenate(
                [self.doc_lengths, np.asarray(new_doc_lengths, dtype=np.float32)]
            )
        else:
            self.doc_lengths = np.asarray(new_doc_lengths, dtype=np.float32)

        self.doc_count = new_count
        self.avgdl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) > 0 else 0.0
        self.recompute_idfs()

    def prune_indices(self, keep_indices: list[int]) -> BM25Index:
        """Construct a new pruned BM25 inverted index retaining only the specified document indices."""
        if not keep_indices:
            return BM25Index(
                doc_lengths=np.zeros(0, dtype=np.float32),
                avgdl=0.0,
                postings={},
                doc_count=0,
                k1=self.k1,
                b=self.b,
            )

        remap: dict[int, int] = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        new_doc_lengths = (
            self.doc_lengths[keep_indices]
            if len(self.doc_lengths) > 0
            else np.zeros(0, dtype=np.float32)
        )
        new_doc_count = len(keep_indices)
        new_avgdl = float(np.mean(new_doc_lengths)) if len(new_doc_lengths) > 0 else 0.0

        new_postings: dict[str, BM25Posting] = {}
        for term, posting in self.postings.items():
            mask = np.isin(posting.doc_indices, keep_indices)
            if not np.any(mask):
                continue
            filtered_doc_ids = posting.doc_indices[mask]
            filtered_freqs = posting.term_freqs[mask]
            remapped_doc_ids = np.array(
                [remap[int(doc_id)] for doc_id in filtered_doc_ids], dtype=np.int32
            )
            n_q = len(remapped_doc_ids)
            idf = float(math.log(1.0 + (new_doc_count - n_q + 0.5) / (n_q + 0.5)))
            new_postings[term] = BM25Posting(
                doc_indices=remapped_doc_ids,
                term_freqs=filtered_freqs,
                idf=idf,
            )

        return BM25Index(
            doc_lengths=new_doc_lengths,
            avgdl=new_avgdl,
            postings=new_postings,
            doc_count=new_doc_count,
            k1=self.k1,
            b=self.b,
        )

    @classmethod
    def build_from_entries(
        cls,
        entries: list[dict[str, Any]],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> BM25Index:
        """Construct an in-memory inverted index and precalculate IDFs from index entries."""
        doc_count = len(entries)
        if doc_count == 0:
            return cls(
                doc_lengths=np.zeros(0, dtype=np.float32),
                avgdl=0.0,
                postings={},
                doc_count=0,
                k1=k1,
                b=b,
            )

        doc_lengths_list: list[float] = []
        raw_postings: dict[str, tuple[list[int], list[float]]] = {}

        for doc_idx, entry in enumerate(entries):
            doc_text = bm25_text_from_entry(entry)
            tokens = tokenize(doc_text)
            doc_length = float(len(tokens))
            doc_lengths_list.append(doc_length)

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term not in raw_postings:
                    raw_postings[term] = ([], [])
                raw_postings[term][0].append(doc_idx)
                raw_postings[term][1].append(float(count))

        doc_lengths = np.asarray(doc_lengths_list, dtype=np.float32)
        avgdl = float(np.mean(doc_lengths)) if len(doc_lengths) > 0 else 0.0

        postings: dict[str, BM25Posting] = {}
        for term, (doc_ids, freqs) in raw_postings.items():
            n_q = len(doc_ids)
            # Robertson-Spärck Jones non-negative IDF
            idf = float(math.log(1.0 + (doc_count - n_q + 0.5) / (n_q + 0.5)))
            postings[term] = BM25Posting(
                doc_indices=np.asarray(doc_ids, dtype=np.int32),
                term_freqs=np.asarray(freqs, dtype=np.float32),
                idf=idf,
            )

        return cls(
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            postings=postings,
            doc_count=doc_count,
            k1=k1,
            b=b,
        )

    def score_tokens(self, query_tokens: tuple[str, ...] | list[str] | set[str]) -> np.ndarray:
        """Calculate vectorized BM25 scores for a set of query tokens."""
        scores = np.zeros(self.doc_count, dtype=np.float32)
        if self.doc_count == 0 or not query_tokens:
            return scores

        unique_tokens = set(query_tokens)
        for term in unique_tokens:
            posting = self.postings.get(term)
            if posting is None:
                continue

            doc_ids = posting.doc_indices
            tf = posting.term_freqs
            idf = posting.idf

            lengths = self.doc_lengths[doc_ids]
            denom = tf + self.k1 * (
                1.0 - self.b + self.b * (lengths / self.avgdl if self.avgdl > 0 else 1.0)
            )
            scores[doc_ids] += idf * (tf * (self.k1 + 1.0)) / denom

        return scores

    def score_query(self, query: str) -> np.ndarray:
        """Tokenize a query string and return BM25 scores across all indexed repositories."""
        if self.doc_count == 0 or not query.strip():
            return np.zeros(self.doc_count, dtype=np.float32)
        tokens = tokenize(query)
        return self.score_tokens(tokens)
