#!/usr/bin/env python3
"""Generate a comprehensive 500k Adversarial/Negative Evaluation benchmark dataset.

The dataset rigorously tests 4 distinct failure modes across 1,000 cases:
1. out_of_domain_no_result: Non-existent, fictional, or physically impossible repositories.
   Expected: System should abstain (abstained=True or zero high-confidence results).
2. alternative_contrastive: Queries asking for alternatives to X (e.g. "open source alternative to redis").
   Forbidden target: The original project itself (e.g. redis/redis). If X is ranked #1, it fails.
3. cross_language_conflict: Queries specifying language A for a tool famous in language B (e.g. "Rust web framework like Django").
   Forbidden target: The original language B repo (e.g. django/django).
4. name_collision_distractor: Exact short names where popular distractors/plugins exist (e.g. searching "vue" should not rank "vue-router" above "vuejs/core").
   Forbidden target: Plugin/distractor repos should not beat the primary canonical repository.
"""

from __future__ import annotations

import json
import random
import resource
import sqlite3
from pathlib import Path
from typing import Any

# Hard memory safety limit: 2.0 GB virtual memory guard
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except (ValueError, OSError):
    pass

DB_PATH = Path("data/index_500k.meta.db")
OUTPUT_PATH = Path("data/eval_cases_negative_500k.json")


def build_negative_eval_dataset(total_target: int = 1000, seed: int = 42) -> dict[str, Any]:
    print(f"Loading repository knowledge from {DB_PATH} (seed={seed})...")
    rng = random.Random(seed)

    conn = sqlite3.connect(f"file:{DB_PATH.resolve().as_posix()}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT repo_id, name, language, stars, entry_json
        FROM records
        WHERE disabled = 0 AND stars >= 200
        ORDER BY stars DESC
        LIMIT 40000
        """
    )

    repo_by_id: dict[str, dict[str, Any]] = {}
    repo_by_name: dict[str, list[str]] = {}
    replaces_pairs: list[dict[str, Any]] = []

    for repo_id, name, language, stars, entry_json in cursor.fetchall():
        name_lower = (name or repo_id.split("/")[-1]).lower()
        lang_lower = (language or "").lower()
        stars_val = stars or 0

        repo_by_id[repo_id] = {
            "repo_id": repo_id,
            "name": name_lower,
            "language": lang_lower,
            "stars": stars_val,
        }
        repo_by_name.setdefault(name_lower, []).append(repo_id)

        try:
            entry = json.loads(entry_json)
            replaces = entry.get("metadata", {}).get("replaces") or []
            for target in replaces:
                target_clean = str(target).strip().lower()
                if len(target_clean) >= 3 and not target_clean.isdigit():
                    replaces_pairs.append(
                        {
                            "candidate_repo_id": repo_id,
                            "candidate_name": name_lower,
                            "candidate_lang": lang_lower,
                            "target_name": target_clean,
                        }
                    )
        except Exception:
            pass

    conn.close()
    print(
        f"Loaded {len(repo_by_id):,} repos, {len(replaces_pairs):,} declared replacement relationships."
    )

    cases: list[dict[str, Any]] = []
    case_idx = 1

    # =========================================================================
    # Type 1: Out-of-domain / Non-existent / Hallucination-trap (Target: 300)
    # =========================================================================
    fictional_technologies = [
        "quantum teleportation protocol in pure brainfuck",
        "warp drive navigation engine simulator for android",
        "flux capacitor time-travel control library for esp32",
        "superconducting cold fusion reactor firmware written in bash",
        "anti-gravity propulsion calculation framework in cobol",
        "telepathic neural lace brain-computer interface driver for dos",
        "tachyon particle physics collision detector in html5",
        "perpetual motion machine energy conservation validator",
        "interdimensional wormhole routing daemon for kubernetes",
        "dark matter gravity wave synthesizer written in vba",
        "zero-point energy harvesting controller for raspberry pi",
        "hyperspace jump coordinates calculating library for commodore 64",
        "molecular nanobot swarm coordination protocol in fortran",
        "faster-than-light subspace communication protocol in lua",
        "astral projection synchronization daemon for systemd",
    ]
    fictional_framework_prefixes = [
        "hyper-quark-db",
        "necro-kernel",
        "chrono-cache",
        "aether-mq",
        "psi-net",
        "sub-atomic-orm",
        "void-compiler",
        "tachyon-rpc",
        "bio-electric-v8",
        "singularity-fs",
        "omni-mesh-core",
        "nebula-flow-ai",
        "zenith-crypto-kv",
        "paragon-matrix-vm",
        "mirage-protocol-x",
    ]
    languages = ["rust", "golang", "python", "typescript", "c++", "zig", "swift", "elixir"]

    print("Generating Type 1: Out-of-domain / No-result trap queries...")
    for text in fictional_technologies:
        for _ in range(12):
            lang = rng.choice(languages)
            cases.append(
                {
                    "id": f"neg-no-result-{case_idx:04d}",
                    "negative_type": "out_of_domain_no_result",
                    "query": f"{text} with {lang}",
                    "expected_repo_id": None,
                    "forbidden_top_repo_ids": [],
                    "expected_behavior": "abstain",
                    "tags": ["negative", "out-of-domain", "no-result", f"language-{lang}"],
                }
            )
            case_idx += 1

    for prefix in fictional_framework_prefixes:
        for lang in languages[:12]:
            cases.append(
                {
                    "id": f"neg-no-result-{case_idx:04d}",
                    "negative_type": "out_of_domain_no_result",
                    "query": f"{prefix}-{lang} framework for distributed cloud synthesis",
                    "expected_repo_id": None,
                    "forbidden_top_repo_ids": [],
                    "expected_behavior": "abstain",
                    "tags": ["negative", "out-of-domain", "no-result", f"language-{lang}"],
                }
            )
            case_idx += 1

    # =========================================================================
    # Type 2: Alternative Contrastive (Target: 300)
    # 期望系统在用户搜索 "alternative to X" 时，绝对不能将 X 原仓库排在 Top 1！
    # =========================================================================
    print("Generating Type 2: Alternative contrastive queries (Forbidden target)...")
    famous_targets = [
        ("redis", ["redis/redis", "antirez/redis"]),
        ("firebase", ["firebase/firebase-js-sdk", "firebase/firebase-ios-sdk"]),
        ("notion", ["notion/notion"]),
        ("docker", ["docker/docker", "moby/moby"]),
        ("postman", ["postmanlabs/postman-app-support"]),
        ("elastic", ["elastic/elasticsearch"]),
        ("elasticsearch", ["elastic/elasticsearch"]),
        ("slack", ["slackapi/python-slack-sdk"]),
        ("trello", ["trello/trello"]),
        ("airtable", ["airtable/airtable.js"]),
        ("kafka", ["apache/kafka"]),
        ("rabbitmq", ["rabbitmq/rabbitmq-server"]),
        ("nginx", ["nginx/nginx"]),
        ("jira", ["atlassian/jira"]),
        ("datadog", ["datadog/dd-trace-py"]),
        ("pagerduty", ["pagerduty/pdpyras"]),
        ("sentry", ["getsentry/sentry"]),
        ("zapier", ["zapier/zapier-platform"]),
        ("stripe", ["stripe/stripe-python", "stripe/stripe-node"]),
        ("terraform", ["hashicorp/terraform"]),
        ("vault", ["hashicorp/vault"]),
        ("consul", ["hashicorp/consul"]),
        ("prometheus", ["prometheus/prometheus"]),
        ("grafana", ["grafana/grafana"]),
    ]

    alt_templates = [
        "open source alternative to {target}",
        "self hosted alternative to {target}",
        "free open-source replacement for {target}",
        "lightweight alternative to {target} in {lang}",
        "drop in replacement for {target}",
        "modern alternative to {target} written in {lang}",
    ]

    for target, forbidden_repos in famous_targets:
        for _ in range(8):
            lang = rng.choice(languages)
            tmpl = rng.choice(alt_templates)
            q = tmpl.format(target=target, lang=lang)
            cases.append(
                {
                    "id": f"neg-alternative-{case_idx:04d}",
                    "negative_type": "alternative_contrastive",
                    "query": q,
                    "expected_repo_id": None,
                    "forbidden_top_repo_ids": forbidden_repos,
                    "target_name": target,
                    "expected_behavior": "target_downranked",
                    "tags": ["negative", "alternative", f"target-{target}"],
                }
            )
            case_idx += 1

    # Add from declared replaces pairs
    for pair in rng.sample(replaces_pairs, min(len(replaces_pairs), 108)):
        target = pair["target_name"]
        lang = pair["candidate_lang"]
        cases.append(
            {
                "id": f"neg-alternative-{case_idx:04d}",
                "negative_type": "alternative_contrastive",
                "query": f"open source alternative to {target} in {lang}"
                if lang
                else f"alternative to {target}",
                "expected_repo_id": pair["candidate_repo_id"],
                "forbidden_top_repo_ids": repo_by_name.get(target, []),
                "target_name": target,
                "expected_behavior": "target_downranked",
                "tags": ["negative", "alternative", f"target-{target}"],
            }
        )
        case_idx += 1

    # =========================================================================
    # Type 3: Cross-Language Conflict (Target: 250)
    # 用户搜索 "Rust web framework like Django"，严禁推荐 django/django！
    # =========================================================================
    print("Generating Type 3: Cross-language conflict queries...")
    famous_frameworks = [
        ("Django", "python", ["django/django"]),
        ("Flask", "python", ["pallets/flask"]),
        ("FastAPI", "python", ["fastapi/fastapi", "tiangolo/fastapi"]),
        ("Tornado", "python", ["tornadoweb/tornado"]),
        ("Sanic", "python", ["sanic-org/sanic"]),
        ("Express", "javascript", ["expressjs/express"]),
        ("Koa", "javascript", ["koajs/koa"]),
        ("NestJS", "typescript", ["nestjs/nest"]),
        ("Fastify", "javascript", ["fastify/fastify"]),
        ("Spring Boot", "java", ["spring-projects/spring-boot"]),
        ("Quarkus", "java", ["quarkusio/quarkus"]),
        ("Gin", "go", ["gin-gonic/gin"]),
        ("Echo", "go", ["labstack/echo"]),
        ("Fiber", "go", ["gofiber/fiber"]),
        ("Actix-web", "rust", ["actix/actix-web"]),
        ("Axum", "rust", ["tokio-rs/axum"]),
        ("Rocket", "rust", ["rwf2/Rocket"]),
        ("Rails", "ruby", ["rails/rails"]),
        ("Sinatra", "ruby", ["sinatra/sinatra"]),
        ("Laravel", "php", ["laravel/laravel"]),
        ("Symfony", "php", ["symfony/symfony"]),
        ("PyTorch", "python", ["pytorch/pytorch"]),
        ("TensorFlow", "python", ["tensorflow/tensorflow"]),
        ("Tokio", "rust", ["tokio-rs/tokio"]),
    ]

    opposing_languages = {
        "python": ["rust", "golang", "c++"],
        "javascript": ["rust", "golang", "python"],
        "typescript": ["rust", "golang", "c++"],
        "java": ["rust", "golang", "python"],
        "go": ["rust", "python", "typescript"],
        "rust": ["python", "golang", "typescript"],
        "ruby": ["golang", "rust", "python"],
        "php": ["python", "golang", "rust"],
    }

    cross_templates = [
        "{req_lang} framework like {orig_name}",
        "{orig_name} equivalent written in {req_lang}",
        "{req_lang} alternative to {orig_name}",
        "similar to {orig_name} but for {req_lang}",
        "fast {req_lang} library inspired by {orig_name}",
    ]

    for orig_name, orig_lang, forbidden_repos in famous_frameworks:
        cand_langs = opposing_languages.get(orig_lang, ["rust", "golang"])
        for req_lang in cand_langs:
            for tmpl in cross_templates:
                q = tmpl.format(req_lang=req_lang, orig_name=orig_name)
                cases.append(
                    {
                        "id": f"neg-cross-lang-{case_idx:04d}",
                        "negative_type": "cross_language_conflict",
                        "query": q,
                        "expected_repo_id": None,
                        "forbidden_top_repo_ids": forbidden_repos,
                        "requested_language": req_lang,
                        "conflicting_language": orig_lang,
                        "expected_behavior": "exclude_forbidden_match_requested_lang",
                        "tags": [
                            "negative",
                            "cross-language",
                            f"requested-{req_lang}",
                            f"original-{orig_lang}",
                        ],
                    }
                )
                case_idx += 1

    # =========================================================================
    # Type 4: Name Collision / Canonical vs Distractor (Target: 150)
    # 搜索 "vue"，Top 1 必须是 vuejs/core 或 vuejs/vue，不能被插件/awesome 列表反超！
    # =========================================================================
    print("Generating Type 4: Exact name collision / distractor queries...")
    canonical_vs_distractors = [
        ("vue", "vuejs/core", ["vuejs/vue-router", "vuejs/vuex", "vuejs/awesome-vue"]),
        ("react", "facebook/react", ["facebook/react-native", "reactjs/react-redux"]),
        ("angular", "angular/angular", ["angular/angular-cli", "angular/components"]),
        ("docker", "moby/moby", ["docker/compose", "docker/cli", "docker/docker-py"]),
        ("kubernetes", "kubernetes/kubernetes", ["kubernetes/client-go", "kubernetes/dashboard"]),
        ("rust", "rust-lang/rust", ["rust-lang/cargo", "rust-lang/book"]),
        ("go", "golang/go", ["golang/tools", "golang/mock"]),
        ("redis", "redis/redis", ["redis/redis-py", "redis/go-redis"]),
        ("neovim", "neovim/neovim", ["neovim/nvim-lspconfig"]),
        ("vite", "vitejs/vite", ["vitejs/vite-plugin-vue"]),
        ("django", "django/django", ["django/django-rest-framework", "django/channels"]),
        ("flask", "pallets/flask", ["pallets/flask-sqlalchemy", "pallets/werkzeug"]),
        ("fastapi", "fastapi/fastapi", ["fastapi/full-stack-fastapi-template"]),
        ("nextjs", "vercel/next.js", ["vercel/commerce", "vercel/examples"]),
        ("tailwind", "tailwindlabs/tailwindcss", ["tailwindlabs/headlessui"]),
    ]

    for name, canonical, distractors in canonical_vs_distractors:
        for q in [
            name,
            f"{name} framework",
            f"{name} library",
            f"github {name}",
            f"get {name}",
            f"{name} official",
        ]:
            cases.append(
                {
                    "id": f"neg-distractor-{case_idx:04d}",
                    "negative_type": "name_collision_distractor",
                    "query": q,
                    "expected_repo_id": canonical,
                    "forbidden_top_repo_ids": distractors,
                    "expected_behavior": "canonical_beats_distractors",
                    "tags": ["negative", "distractor-defense", f"name-{name}"],
                }
            )
            case_idx += 1

    rng.shuffle(cases)
    cases = cases[:total_target]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    type_counts = {}
    for c in cases:
        t = c["negative_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    summary = {
        "total_cases": len(cases),
        "types": type_counts,
        "output_path": str(OUTPUT_PATH),
    }
    print(f"\nSuccessfully generated {len(cases):,} adversarial negative cases!")
    print(f"Type Breakdown: {type_counts}")
    print(f"Saved to: {OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    summary = build_negative_eval_dataset(total_target=1000)
    print("\nGeneration Summary:", json.dumps(summary, indent=2))
