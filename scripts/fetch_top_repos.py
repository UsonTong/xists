#!/usr/bin/env python3
"""Fetch Top GitHub Repository Names with Multi-Token Rotation and Dynamic Star Slicing.

Outputs:
  - Text file with one owner/repo per line (e.g., data/top100k_repos.txt)
  - Metadata JSONL with repo_id, stargazers_count, language, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable
from math import ceil
from pathlib import Path
from typing import Any

USER_AGENT = "xists-top-repo-fetcher/1.0"
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse key=value pairs from a .env file if it exists."""
    env_vars: dict[str, str] = {}
    if not path.exists():
        return env_vars
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip("'\"")
        env_vars[key.strip()] = val
    return env_vars


def resolve_tokens(explicit_tokens: list[str] | None = None) -> list[str]:
    """Resolve GitHub tokens from arguments, environment, or .env file."""
    if explicit_tokens:
        return [t.strip() for t in explicit_tokens if t.strip()]

    # Check process environment first
    tokens_raw = os.environ.get("GITHUB_TOKENS", "").strip()
    if tokens_raw:
        return [t.strip() for t in tokens_raw.split(",") if t.strip()]
    single = os.environ.get("GITHUB_TOKEN", "").strip()
    if single:
        return [single]

    # Check .env in current directory or repo root
    for candidate in [Path(".env"), Path(__file__).resolve().parents[1] / ".env"]:
        env_dict = _load_env_file(candidate)
        if "GITHUB_TOKENS" in env_dict and env_dict["GITHUB_TOKENS"]:
            return [t.strip() for t in env_dict["GITHUB_TOKENS"].split(",") if t.strip()]
        if "GITHUB_TOKEN" in env_dict and env_dict["GITHUB_TOKEN"]:
            return [env_dict["GITHUB_TOKEN"].strip()]

    return []


class SearchTokenPool:
    """Round-robin token pool managing GitHub Search API rate limits."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens or [""]  # empty string allows unauthenticated calls
        self._index = 0
        self._lock = threading.Lock()
        self._rate_limited_until: dict[str, float] = {}

    def get_token(self) -> str:
        with self._lock:
            now = time.time()
            for _ in range(len(self.tokens)):
                token = self.tokens[self._index % len(self.tokens)]
                self._index += 1
                if self._rate_limited_until.get(token, 0.0) <= now:
                    self._rate_limited_until.pop(token, None)
                    return token
            # All tokens are currently limited; wait for the earliest reset
            earliest_reset = min(self._rate_limited_until.values())
            wait_time = max(1.0, earliest_reset - now + 1.0)
            token = min(self._rate_limited_until, key=lambda k: self._rate_limited_until[k])
            print(
                f"\n[!] All GitHub tokens search-limited. Sleeping {wait_time:.1f}s for reset...",
                flush=True,
            )
            time.sleep(wait_time)
            self._rate_limited_until.pop(token, None)
            return token

    def mark_rate_limited(self, token: str, reset_timestamp: float) -> None:
        with self._lock:
            if token:
                self._rate_limited_until[token] = reset_timestamp


def _search_github_api(
    query: str,
    page: int,
    per_page: int,
    token_pool: SearchTokenPool,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Execute a GitHub search repository query with token rotation and rate limit handling."""
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": str(per_page),
        "page": str(page),
    }
    url = f"{GITHUB_SEARCH_API}?{urllib.parse.urlencode(params)}"

    for attempt in range(max_retries):
        token = token_pool.get_token()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # Check remaining rate limit from headers
                remaining = resp.headers.get("x-ratelimit-remaining")
                reset = resp.headers.get("x-ratelimit-reset")
                if remaining == "0" and reset:
                    token_pool.mark_rate_limited(token, float(reset))
                return data
        except urllib.error.HTTPError as err:
            reset = err.headers.get("x-ratelimit-reset")
            if err.code in {403, 429}:
                reset_ts = float(reset) if reset else time.time() + 60.0
                token_pool.mark_rate_limited(token, reset_ts)
                wait_sec = min(30.0, max(2.0, reset_ts - time.time() + 1.0))
                time.sleep(wait_sec)
                continue
            if err.code in {500, 502, 503, 504}:
                time.sleep(2**attempt)
                continue
            # For 422 (validation/query parse error), return empty
            if err.code == 422:
                return {"total_count": 0, "items": []}
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**attempt)
            continue

    return {"total_count": 0, "items": []}


def fetch_top_repos_by_star_slicing(
    target_count: int,
    token_pool: SearchTokenPool,
    min_stars: int = 150,
    checkpoint_path: Path | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Recursively slice Star count ranges to bypass GitHub Search 1,000 item limit."""
    collected: dict[str, dict[str, Any]] = {}

    # Resume from checkpoint if present
    if checkpoint_path and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                repo_id = rec.get("full_name") or rec.get("repo_id")
                if repo_id:
                    collected[repo_id.lower()] = rec
            except Exception:
                pass
        print(f"[*] Resumed {len(collected):,} repositories from checkpoint: {checkpoint_path}")

    # Queue of star ranges: (min_stars, max_stars)
    # Start with highest stars (500k) down to min_stars
    queue: deque[tuple[int, int]] = deque(
        [
            (100000, 500000),
            (50000, 99999),
            (30000, 49999),
            (20000, 29999),
            (15000, 19999),
            (10000, 14999),
            (8000, 9999),
            (6000, 7999),
            (4000, 5999),
            (3000, 3999),
            (2000, 2999),
            (1500, 1999),
            (1000, 1499),
            (750, 999),
            (500, 749),
            (350, 499),
            (250, 349),
            (180, 249),
            (min_stars, 179),
        ]
    )

    checkpoint_file = None
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path.open("a", encoding="utf-8")

    try:
        while queue and len(collected) < target_count:
            low, high = queue.popleft()
            if high < min_stars:
                continue

            query = f"stars:{low}..{high} fork:false"
            initial_resp = _search_github_api(query, page=1, per_page=100, token_pool=token_pool)
            total_count = initial_resp.get("total_count", 0)

            if total_count == 0:
                continue

            # If count > 1000, GitHub Search won't return items past page 10 (1000). Bisect!
            if total_count > 1000 and high > low:
                mid = (low + high) // 2
                # High half first so we fetch higher stars first
                queue.appendleft((low, mid))
                queue.appendleft((mid + 1, high))
                continue

            # If total_count <= 1000 (or cannot bisect further), fetch all available pages
            pages = min(10, ceil(total_count / 100))
            for page in range(1, pages + 1):
                if len(collected) >= target_count:
                    break

                if page == 1:
                    data = initial_resp
                else:
                    data = _search_github_api(query, page=page, per_page=100, token_pool=token_pool)

                items = data.get("items", [])
                if not items:
                    break

                new_count = 0
                for item in items:
                    name = item.get("full_name", "")
                    if not name:
                        continue
                    key = name.lower()
                    if key not in collected:
                        record = {
                            "repo_id": name,
                            "full_name": name,
                            "stargazers_count": item.get("stargazers_count", 0),
                            "forks_count": item.get("forks_count", 0),
                            "language": item.get("language"),
                            "description": item.get("description"),
                            "topics": item.get("topics", []),
                            "html_url": item.get("html_url"),
                            "created_at": item.get("created_at"),
                            "updated_at": item.get("updated_at"),
                        }
                        collected[key] = record
                        new_count += 1
                        if checkpoint_file:
                            checkpoint_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            checkpoint_file.flush()

                if on_progress:
                    on_progress(len(collected), target_count, low)

            # Small delay between ranges to be polite
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(
            f"\n[!] Interrupted by user! Gracefully preserving {len(collected):,} collected repositories...",
            flush=True,
        )
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Top 100k GitHub Repository Names with Token Rotation."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100000,
        help="Target number of unique repositories to collect (default: 100,000)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=100,
        help="Minimum star count threshold (default: 100)",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("data/top100k_repos.txt"),
        help="Output plain text file path (default: data/top100k_repos.txt)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/top100k_repos.jsonl"),
        help="Output metadata JSONL file path (default: data/top100k_repos.jsonl)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/top100k_repos.partial.jsonl"),
        help="Intermediate checkpoint file for resume",
    )
    parser.add_argument(
        "--tokens",
        nargs="*",
        help="Optional explicit list of GitHub tokens",
    )

    args = parser.parse_args()

    tokens = resolve_tokens(args.tokens)
    print("=== xists Top GitHub Repositories Collector ===")
    print(f"Target count:       {args.target:,}")
    print(f"Min stars:          {args.min_stars:,}")
    print(f"Loaded tokens:      {len(tokens)} token(s) configured")
    print(f"Output text path:   {args.output_txt}")
    print(f"Output JSONL path:  {args.output_jsonl}")
    print(f"Checkpoint path:    {args.checkpoint}")
    print("=" * 48)

    token_pool = SearchTokenPool(tokens)

    start_time = time.time()
    last_print = 0.0

    def progress_callback(current: int, total: int, current_star_bound: int) -> None:
        nonlocal last_print
        now = time.time()
        if now - last_print >= 0.5 or current >= total:
            last_print = now
            elapsed = now - start_time
            qps = current / elapsed if elapsed > 0 else 0
            pct = (current / total) * 100.0 if total > 0 else 0
            sys.stdout.write(
                f"\r[*] Progress: {current:,}/{total:,} repos ({pct:.1f}%) | "
                f"Stars boundary: ≥{current_star_bound:,} | "
                f"Rate: {qps:.1f} repos/s | Elapsed: {elapsed:.0f}s"
            )
            sys.stdout.flush()

    repos_dict = fetch_top_repos_by_star_slicing(
        target_count=args.target,
        token_pool=token_pool,
        min_stars=args.min_stars,
        checkpoint_path=args.checkpoint,
        on_progress=progress_callback,
    )

    print("\n\n[*] Collection completed! Sorting by stars descending...")

    # Sort all collected repositories by star count descending
    sorted_repos = sorted(
        repos_dict.values(),
        key=lambda r: int(r.get("stargazers_count", 0)),
        reverse=True,
    )

    # Ensure output directories exist
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Write plain text repo list (one repo per line: owner/repo)
    with args.output_txt.open("w", encoding="utf-8") as f_txt:
        for repo in sorted_repos:
            f_txt.write(f"{repo['repo_id']}\n")

    # Write JSONL metadata
    with args.output_jsonl.open("w", encoding="utf-8") as f_jsonl:
        for rank, repo in enumerate(sorted_repos, 1):
            repo_copy = dict(repo)
            repo_copy["rank"] = rank
            f_jsonl.write(json.dumps(repo_copy, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time
    print(f"[✓] Successfully saved {len(sorted_repos):,} unique repositories!")
    print(f"    - Text list: {args.output_txt} ({args.output_txt.stat().st_size / 1024:.1f} KB)")
    print(
        f"    - JSONL metadata: {args.output_jsonl} ({args.output_jsonl.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    if sorted_repos:
        print(
            f"    - Highest stars: {sorted_repos[0]['repo_id']} ({sorted_repos[0]['stargazers_count']:,} stars)"
        )
        print(
            f"    - Lowest stars:  {sorted_repos[-1]['repo_id']} ({sorted_repos[-1]['stargazers_count']:,} stars)"
        )
    print(f"    - Total duration: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
