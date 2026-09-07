"""Composite boolean facet expression parser and bitmask filter compiler for xists."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from xists.search.query import PreparedIndex
    from xists.types import SearchFilter


class FacetParseError(ValueError):
    """Raised when a boolean facet query expression cannot be parsed."""


# ---------------------------------------------------------------------------
# AST Node Definitions
# ---------------------------------------------------------------------------


class FacetNode:
    """Base class for all facet filter AST nodes."""


@dataclass(frozen=True)
class AndNode(FacetNode):
    """Logical AND of multiple facet sub-expressions."""

    children: Sequence[FacetNode]


@dataclass(frozen=True)
class OrNode(FacetNode):
    """Logical OR of multiple facet sub-expressions."""

    children: Sequence[FacetNode]


@dataclass(frozen=True)
class NotNode(FacetNode):
    """Logical NOT of a facet sub-expression."""

    child: FacetNode


@dataclass(frozen=True)
class PredicateNode(FacetNode):
    """Leaf predicate evaluating a field condition (e.g. stars > 1000, lang == python)."""

    field: str
    operator: str  # "eq", "ne", "gt", "gte", "lt", "lte", "range", "contains"
    value: Any


# ---------------------------------------------------------------------------
# Lexer & Tokenizer
# ---------------------------------------------------------------------------

_OPERATORS = {"AND", "OR", "NOT", "&&", "||", "!", "-", "&", "|"}

_NUMERIC_SUFFIX_MAP = {
    "k": 1_000,
    "m": 1_000_000,
    "g": 1_000_000_000,
}


def _parse_numeric_value(val_str: str) -> int | float:
    """Parse integer or float with optional k/m/g suffix (e.g., '10k' -> 10000)."""
    cleaned = val_str.replace(",", "").strip().lower()
    for suffix, multiplier in _NUMERIC_SUFFIX_MAP.items():
        if cleaned.endswith(suffix):
            num_part = cleaned[: -len(suffix)].strip()
            return int(float(num_part) * multiplier)
    if "." in cleaned:
        return float(cleaned)
    return int(cleaned)


@dataclass(frozen=True)
class Token:
    kind: str  # 'LPAREN', 'RPAREN', 'AND', 'OR', 'NOT', 'TERM'
    value: str
    pos: int


def tokenize_facet_query(query: str) -> list[Token]:
    """Tokenize a composite boolean facet expression string."""
    tokens: list[Token] = []
    i = 0
    n = len(query)

    while i < n:
        char = query[i]

        if char.isspace():
            i += 1
            continue

        if char == "(":
            tokens.append(Token("LPAREN", "(", i))
            i += 1
            continue

        if char == ")":
            tokens.append(Token("RPAREN", ")", i))
            i += 1
            continue

        # Check two-character operators
        if i + 1 < n:
            two_chars = query[i : i + 2]
            if two_chars == "&&":
                tokens.append(Token("AND", "&&", i))
                i += 2
                continue
            if two_chars == "||":
                tokens.append(Token("OR", "||", i))
                i += 2
                continue

        # Single character operators ! and -
        if char == "!":
            tokens.append(Token("NOT", "!", i))
            i += 1
            continue

        if char == "-":
            # If immediately followed by a field/term without space, it's NOT
            tokens.append(Token("NOT", "-", i))
            i += 1
            continue

        if char == "&":
            tokens.append(Token("AND", "&", i))
            i += 1
            continue

        if char == "|":
            tokens.append(Token("OR", "|", i))
            i += 1
            continue

        # Read term or quoted string or field:value
        start_pos = i
        term_chars: list[str] = []
        in_quotes = False
        quote_char = ""

        while i < n:
            c = query[i]
            if c in ('"', "'"):
                if in_quotes and c == quote_char:
                    in_quotes = False
                    quote_char = ""
                    term_chars.append(c)
                    i += 1
                elif not in_quotes:
                    in_quotes = True
                    quote_char = c
                    term_chars.append(c)
                    i += 1
                else:
                    term_chars.append(c)
                    i += 1
            elif in_quotes:
                term_chars.append(c)
                i += 1
            else:
                if c.isspace() or c in ("(", ")"):
                    break
                if c in ("&", "|", "!") and term_chars:
                    break
                term_chars.append(c)
                i += 1

        term_str = "".join(term_chars)
        upper_term = term_str.upper()

        if upper_term == "AND":
            tokens.append(Token("AND", term_str, start_pos))
        elif upper_term == "OR":
            tokens.append(Token("OR", term_str, start_pos))
        elif upper_term == "NOT":
            tokens.append(Token("NOT", term_str, start_pos))
        else:
            tokens.append(Token("TERM", term_str, start_pos))

    return tokens


# ---------------------------------------------------------------------------
# Predicate Parser
# ---------------------------------------------------------------------------


def _strip_quotes(value: str) -> str:
    cleaned = value.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        return cleaned[1:-1]
    return cleaned


def _parse_term_predicate(term: str) -> FacetNode:
    """Parse a single facet term (e.g. 'lang:python', 'stars:>1000', 'is:archived') into AST."""
    term = term.strip()
    if not term:
        raise FacetParseError("Empty facet predicate term")

    # 1. Flag predicates (is:archived, not:archived, is:disabled, not:disabled)
    low_term = term.lower()
    if low_term in ("is:archived", "archived:true"):
        return PredicateNode(field="archived", operator="eq", value=True)
    if low_term in ("not:archived", "is:not_archived", "archived:false"):
        return PredicateNode(field="archived", operator="eq", value=False)
    if low_term in ("is:disabled", "disabled:true"):
        return PredicateNode(field="disabled", operator="eq", value=True)
    if low_term in ("not:disabled", "is:not_disabled", "disabled:false"):
        return PredicateNode(field="disabled", operator="eq", value=False)

    # 2. Key-Value or Key-Operator-Value syntax (e.g. field:value, field=value, field>value)
    match = re.match(r"^([a-zA-Z0-9_-]+)\s*(:|>=|<=|>|<|==|=|\!=)\s*(.+)$", term)
    if match:
        field_raw, op_raw, val_raw = match.groups()
        field = field_raw.lower()
        val_str = _strip_quotes(val_raw)

        # Map field aliases
        if field in ("lang", "language"):
            field = "language"
        elif field in ("eco", "ecosystem"):
            field = "ecosystem"
        elif field in ("type", "project_type"):
            field = "project_type"
        elif field in ("star", "stars"):
            field = "stars"
        elif field in ("fork", "forks"):
            field = "forks"
        elif field in ("topic", "topics", "tag", "tags"):
            field = "topics"
        elif field in ("lic", "license"):
            field = "license"
        elif field in ("repo", "repo_id"):
            field = "repo_id"
        elif field in ("name",):
            field = "name"

        # Check numeric field operations (stars, forks)
        if field in ("stars", "forks"):
            if ".." in val_str:
                parts = val_str.split("..", 1)
                min_v = int(_parse_numeric_value(parts[0]))
                max_v = int(_parse_numeric_value(parts[1]))
                return PredicateNode(field=field, operator="range", value=(min_v, max_v))

            if op_raw in (">", ">=", "<", "<=", "==", "=", "!="):
                op_map = {
                    ">": "gt",
                    ">=": "gte",
                    "<": "lt",
                    "<=": "lte",
                    "==": "eq",
                    "=": "eq",
                    "!=": "ne",
                }
                return PredicateNode(
                    field=field,
                    operator=op_map[op_raw],
                    value=int(_parse_numeric_value(val_str)),
                )

            # op_raw is ':'
            if val_str.startswith(">="):
                return PredicateNode(
                    field=field,
                    operator="gte",
                    value=int(_parse_numeric_value(val_str[2:])),
                )
            if val_str.startswith(">"):
                return PredicateNode(
                    field=field,
                    operator="gt",
                    value=int(_parse_numeric_value(val_str[1:])),
                )
            if val_str.startswith("<="):
                return PredicateNode(
                    field=field,
                    operator="lte",
                    value=int(_parse_numeric_value(val_str[2:])),
                )
            if val_str.startswith("<"):
                return PredicateNode(
                    field=field,
                    operator="lt",
                    value=int(_parse_numeric_value(val_str[1:])),
                )
            if val_str.startswith("="):
                return PredicateNode(
                    field=field,
                    operator="eq",
                    value=int(_parse_numeric_value(val_str[1:])),
                )

            # Bare number under stars: e.g. stars:1000 default to gte
            return PredicateNode(
                field=field,
                operator="gte",
                value=int(_parse_numeric_value(val_str)),
            )

        # Boolean flags via field:true/false
        if field in ("archived", "disabled"):
            is_true = val_str.lower() in ("true", "1", "yes")
            return PredicateNode(field=field, operator="eq", value=is_true)

        # String fields
        return PredicateNode(field=field, operator="contains", value=val_str)

    # Bare term without colon/operator: treat as topic or keyword filter
    return PredicateNode(field="topics", operator="contains", value=_strip_quotes(term))


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------


class FacetParser:
    """Parses token streams into a boolean FacetNode AST with operator precedence."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> FacetNode | None:
        if not self.tokens:
            return None
        expr = self.parse_or()
        if self.pos < len(self.tokens):
            remaining = self.tokens[self.pos]
            raise FacetParseError(
                f"Unexpected token {remaining.value!r} at character position {remaining.pos}"
            )
        return expr

    def parse_or(self) -> FacetNode:
        children = [self.parse_and()]
        while self.current() and self.current().kind == "OR":  # type: ignore[union-attr]
            self.advance()  # consume OR
            children.append(self.parse_and())
        if len(children) == 1:
            return children[0]
        return OrNode(children=children)

    def parse_and(self) -> FacetNode:
        children = [self.parse_unary()]
        while self.current():
            curr = self.current()
            assert curr is not None
            if curr.kind == "AND":
                self.advance()  # consume explicit AND
                children.append(self.parse_unary())
            elif curr.kind in ("NOT", "TERM", "LPAREN"):
                # Implicit AND juxtaposition
                children.append(self.parse_unary())
            else:
                break
        if len(children) == 1:
            return children[0]
        return AndNode(children=children)

    def parse_unary(self) -> FacetNode:
        curr = self.current()
        if not curr:
            raise FacetParseError("Unexpected end of expression, expected facet term or group")

        if curr.kind == "NOT":
            self.advance()  # consume NOT
            child = self.parse_unary()
            return NotNode(child=child)

        return self.parse_primary()

    def parse_primary(self) -> FacetNode:
        curr = self.current()
        if not curr:
            raise FacetParseError("Unexpected end of expression, expected facet term")

        if curr.kind == "LPAREN":
            self.advance()  # consume (
            expr = self.parse_or()
            close_tok = self.current()
            if not close_tok or close_tok.kind != "RPAREN":
                pos = close_tok.pos if close_tok else len(self.tokens)
                raise FacetParseError(f"Unclosed parenthesis at position {pos}")
            self.advance()  # consume )
            return expr

        if curr.kind == "TERM":
            self.advance()  # consume TERM
            return _parse_term_predicate(curr.value)

        raise FacetParseError(
            f"Unexpected token {curr.value!r} of type {curr.kind} at position {curr.pos}"
        )


def parse_facet_query(query: str) -> FacetNode | None:
    """Parse a boolean facet query string into an AST."""
    cleaned = query.strip()
    if not cleaned:
        return None
    tokens = tokenize_facet_query(cleaned)
    parser = FacetParser(tokens)
    return parser.parse()


def parse_filter_criteria(
    filters: SearchFilter | dict[str, Any] | str | None,
) -> FacetNode | None:
    """Normalize and convert SearchFilter dict or expression string into a unified FacetNode AST."""
    if filters is None:
        return None

    if isinstance(filters, str):
        return parse_facet_query(filters)

    if not isinstance(filters, dict):
        return None

    # Check if any filter condition is actually active
    has_active = False
    for k, v in filters.items():
        if v is not None and v is not False:
            has_active = True
            break
        if k == "include_archived" and v is False:
            has_active = True
            break
    if not has_active:
        return None

    nodes: list[FacetNode] = []

    # 1. If explicit query expression string is present
    for expr_key in ("expr", "query", "filter"):
        expr_val = filters.get(expr_key)
        if isinstance(expr_val, str) and expr_val.strip():
            parsed_expr = parse_facet_query(expr_val)
            if parsed_expr is not None:
                nodes.append(parsed_expr)

    # 2. Structured facet keys
    lang_val = filters.get("language")
    if isinstance(lang_val, str) and lang_val.strip():
        nodes.append(PredicateNode(field="language", operator="contains", value=lang_val.strip()))

    eco_val = filters.get("ecosystem")
    if isinstance(eco_val, str) and eco_val.strip():
        nodes.append(PredicateNode(field="ecosystem", operator="contains", value=eco_val.strip()))
    elif isinstance(eco_val, (list, tuple, set)):
        eco_nodes = [
            PredicateNode(field="ecosystem", operator="contains", value=str(e).strip())
            for e in eco_val
            if str(e).strip()
        ]
        if len(eco_nodes) == 1:
            nodes.append(eco_nodes[0])
        elif len(eco_nodes) > 1:
            nodes.append(OrNode(children=eco_nodes))

    pt_val = filters.get("project_type")
    if isinstance(pt_val, str) and pt_val.strip():
        nodes.append(PredicateNode(field="project_type", operator="contains", value=pt_val.strip()))

    min_stars = filters.get("min_stars")
    if min_stars is not None:
        nodes.append(PredicateNode(field="stars", operator="gte", value=int(min_stars)))

    max_stars = filters.get("max_stars")
    if max_stars is not None:
        nodes.append(PredicateNode(field="stars", operator="lte", value=int(max_stars)))

    lic_val = filters.get("license")
    if isinstance(lic_val, str) and lic_val.strip():
        nodes.append(PredicateNode(field="license", operator="contains", value=lic_val.strip()))

    topics_val = filters.get("topics")
    if isinstance(topics_val, str) and topics_val.strip():
        nodes.append(PredicateNode(field="topics", operator="contains", value=topics_val.strip()))
    elif isinstance(topics_val, (list, tuple, set)):
        for t in topics_val:
            if str(t).strip():
                nodes.append(
                    PredicateNode(field="topics", operator="contains", value=str(t).strip())
                )

    include_archived = bool(filters.get("include_archived", False))
    if not include_archived:
        nodes.append(NotNode(child=PredicateNode(field="archived", operator="eq", value=True)))

    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return AndNode(children=nodes)


# ---------------------------------------------------------------------------
# AST Compilation to Vectorized NumPy Bitmasks
# ---------------------------------------------------------------------------


def evaluate_ast_mask(node: FacetNode, prepared: PreparedIndex) -> np.ndarray:
    """Compile and evaluate a FacetNode AST into a 1D boolean NumPy bitmask over a PreparedIndex."""
    doc_count = len(prepared.entries)
    if doc_count == 0:
        return np.empty(0, dtype=bool)

    if isinstance(node, AndNode):
        if not node.children:
            return np.ones(doc_count, dtype=bool)
        mask = evaluate_ast_mask(node.children[0], prepared)
        for child in node.children[1:]:
            mask = mask & evaluate_ast_mask(child, prepared)
        return mask

    if isinstance(node, OrNode):
        if not node.children:
            return np.zeros(doc_count, dtype=bool)
        mask = evaluate_ast_mask(node.children[0], prepared)
        for child in node.children[1:]:
            mask = mask | evaluate_ast_mask(child, prepared)
        return mask

    if isinstance(node, NotNode):
        return ~evaluate_ast_mask(node.child, prepared)

    if isinstance(node, PredicateNode):
        field = node.field
        op = node.operator
        val = node.value

        # Numeric stars field
        if field == "stars":
            stars = prepared.stars_array
            if op == "gt":
                return stars > val
            if op == "gte":
                return stars >= val
            if op == "lt":
                return stars < val
            if op == "lte":
                return stars <= val
            if op == "eq":
                return stars == val
            if op == "ne":
                return stars != val
            if op == "range" and isinstance(val, (tuple, list)):
                return (stars >= val[0]) & (stars <= val[1])
            return stars >= val

        # Numeric forks field
        if field == "forks":
            forks = prepared.forks_array
            if op == "gt":
                return forks > val
            if op == "gte":
                return forks >= val
            if op == "lt":
                return forks < val
            if op == "lte":
                return forks <= val
            if op == "eq":
                return forks == val
            if op == "ne":
                return forks != val
            if op == "range" and isinstance(val, (tuple, list)):
                return (forks >= val[0]) & (forks <= val[1])
            return forks >= val

        # Boolean archived / disabled field
        if field in ("archived", "disabled"):
            archived = prepared.archived_array
            return archived if val is True else ~archived

        # Language field
        if field == "language":
            from xists.search.query import LANGUAGE_ALIASES, _metadata_language_alias

            target_lang = str(val).strip().lower()
            target_canonical = _metadata_language_alias(target_lang)
            target_aliases = (
                LANGUAGE_ALIASES.get(target_canonical, set()) if target_canonical else set()
            )
            return np.array(
                [
                    bool(
                        (target_canonical and c["language_alias"] == target_canonical)
                        or (c["language_lower"] in target_aliases)
                        or (target_lang == c["language_lower"])
                        or (not target_canonical and target_lang in c["language_lower"])
                    )
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

        # Ecosystem field
        if field == "ecosystem":
            target_eco = str(val).strip().lower()
            return np.array(
                [
                    bool(
                        target_eco in c["ecosystem_set"]
                        or any(target_eco in e for e in c["ecosystem_set"])
                    )
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

        # Project type field
        if field == "project_type":
            target_pt = str(val).strip().lower().replace("-", "_").replace(" ", "_")
            return np.array(
                [
                    bool(
                        c["project_type_norm"]
                        and (
                            c["project_type_norm"] == target_pt
                            or target_pt in c["project_type_norm"]
                        )
                    )
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

        # Topics field
        if field == "topics":
            target_topic = str(val).strip().lower()
            return np.array(
                [
                    bool(
                        target_topic in c["topics_set"]
                        or any(target_topic in t for t in c["topics_set"])
                    )
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

        # License field
        if field == "license":
            target_lic = str(val).strip().lower()
            return np.array(
                [
                    bool(
                        c["license_lower"]
                        and (c["license_lower"] == target_lic or target_lic in c["license_lower"])
                    )
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

        # Repo ID field
        if field == "repo_id":
            target_repo = str(val).strip().lower()
            return np.array(
                [
                    bool(c["repo_id_lower"] == target_repo or target_repo in c["repo_id_lower"])
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

        # Name field
        if field == "name":
            target_name = str(val).strip().lower()
            return np.array(
                [
                    bool(
                        target_name in c["identity_values_lower"]
                        or target_name == c["repo_id_lower"].split("/")[-1]
                    )
                    for c in prepared.metadata_caches
                ],
                dtype=bool,
            )

    return np.ones(doc_count, dtype=bool)


def evaluate_ast_record(node: FacetNode, record: dict[str, Any]) -> bool:
    """Evaluate a FacetNode AST against a single repository record or metadata dictionary."""
    if isinstance(node, AndNode):
        return all(evaluate_ast_record(child, record) for child in node.children)

    if isinstance(node, OrNode):
        return any(evaluate_ast_record(child, record) for child in node.children)

    if isinstance(node, NotNode):
        return not evaluate_ast_record(node.child, record)

    if isinstance(node, PredicateNode):
        raw_github = record.get("github")
        github: dict[str, Any] = raw_github if isinstance(raw_github, dict) else {}
        raw_meta = record.get("metadata")
        metadata: dict[str, Any] = (
            raw_meta if isinstance(raw_meta, dict) else record if "stars" in record else {}
        )
        raw_profile = record.get("llm_profile")
        profile: dict[str, Any] = raw_profile if isinstance(raw_profile, dict) else {}

        field = node.field
        op = node.operator
        val = node.value

        if field == "stars":
            stars = int(metadata.get("stars") or github.get("stars") or 0)
            if op == "gt":
                return stars > val
            if op == "gte":
                return stars >= val
            if op == "lt":
                return stars < val
            if op == "lte":
                return stars <= val
            if op == "eq":
                return stars == val
            if op == "ne":
                return stars != val
            if op == "range" and isinstance(val, (tuple, list)):
                return val[0] <= stars <= val[1]
            return stars >= val

        if field == "forks":
            forks = int(metadata.get("forks") or github.get("forks") or 0)
            if op == "gt":
                return forks > val
            if op == "gte":
                return forks >= val
            if op == "lt":
                return forks < val
            if op == "lte":
                return forks <= val
            if op == "eq":
                return forks == val
            if op == "ne":
                return forks != val
            if op == "range" and isinstance(val, (tuple, list)):
                return val[0] <= forks <= val[1]
            return forks >= val

        if field in ("archived", "disabled"):
            is_archived = bool(
                metadata.get("archived")
                or github.get("archived")
                or metadata.get("disabled")
                or github.get("disabled")
            )
            return is_archived if val is True else not is_archived

        if field == "language":
            from xists.search.query import LANGUAGE_ALIASES, _metadata_language_alias

            lang = str(metadata.get("language") or github.get("language") or "").lower().strip()
            target_lang = str(val).strip().lower()
            target_canonical = _metadata_language_alias(target_lang)
            target_aliases = (
                LANGUAGE_ALIASES.get(target_canonical, set()) if target_canonical else set()
            )
            return bool(
                (target_canonical and _metadata_language_alias(lang) == target_canonical)
                or (lang in target_aliases)
                or (target_lang == lang)
                or (not target_canonical and target_lang in lang)
            )

        if field == "ecosystem":
            target_eco = str(val).strip().lower()
            ecos = [
                str(e).lower()
                for e in (metadata.get("ecosystem") or profile.get("ecosystem") or [])
            ]
            return any(target_eco in e for e in ecos)

        if field == "project_type":
            target_pt = str(val).strip().lower().replace("-", "_").replace(" ", "_")
            pt = str(metadata.get("project_type") or profile.get("project_type") or "").lower()
            return target_pt in pt

        if field == "topics":
            target_topic = str(val).strip().lower()
            topics = [
                str(t).lower() for t in (metadata.get("topics") or github.get("topics") or [])
            ]
            return any(target_topic in t for t in topics)

        if field == "license":
            target_lic = str(val).strip().lower()
            lic = str(metadata.get("license") or github.get("license") or "").lower()
            return target_lic in lic

        if field == "repo_id":
            target_repo = str(val).strip().lower()
            rid = str(record.get("repo_id") or "").lower()
            return target_repo in rid

        if field == "name":
            target_name = str(val).strip().lower()
            name = str(record.get("name") or metadata.get("name") or "").lower()
            return target_name in name

    return True
