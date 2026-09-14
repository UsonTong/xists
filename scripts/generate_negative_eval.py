"""Generate a high-quality negative/contrastive evaluation benchmark dataset.

The dataset tests 4 distinct failure modes that pure BM25 or naive search fails on:
1. out_of_domain_no_result: Non-existent, fictional, or physically impossible repositories.
   Expected behavior: System should abstain (abstained=True or zero high-confidence results).
2. alternative_contrastive: Queries asking for alternatives to X (e.g. "open source alternative to redis").
   Forbidden target: The original project itself (e.g. redis/redis). If X is ranked #1, it fails.
3. cross_language_conflict: Queries specifying language A for a tool famous in language B (e.g. "Rust web framework like express").
   Forbidden target: The original language B repo (e.g. expressjs/express).
4. name_collision_distractor: Exact short names where popular distractors/plugins exist (e.g. searching "vue" should not rank "vue-router" or "awesome-vue" above "vuejs/core").
   Forbidden target: Plugin/distractor repos should not beat the primary canonical repository.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def build_negative_eval_dataset(
    records_path: Path = Path("data/records_100k.jsonl"),
    output_path: Path = Path("data/eval_cases_negative_1k.json"),
    total_target: int = 1000,
) -> dict[str, Any]:
    print(f"Loading repository knowledge from {records_path}...")

    repo_by_id: dict[str, dict[str, Any]] = {}
    repo_by_name: dict[str, list[str]] = {}
    replaces_pairs: list[dict[str, Any]] = []

    with open(records_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 30000:  # Sample from top 30k repositories
                break
            r = json.loads(line)
            rid = r.get("repo_id")
            if not rid:
                continue
            name = (r.get("name") or rid.split("/")[-1]).lower()
            github = r.get("github") or {}
            prof = r.get("llm_profile") or {}
            lang = (github.get("language") or "").lower()
            stars = github.get("stars", 0)

            repo_by_id[rid] = {
                "repo_id": rid,
                "name": name,
                "language": lang,
                "stars": stars,
                "description": github.get("description") or "",
            }
            repo_by_name.setdefault(name, []).append(rid)

            replaces = prof.get("replaces") or []
            for target in replaces:
                target_clean = str(target).strip().lower()
                if len(target_clean) >= 3 and not target_clean.isdigit():
                    replaces_pairs.append(
                        {
                            "candidate_repo_id": rid,
                            "candidate_name": name,
                            "candidate_lang": lang,
                            "target_name": target_clean,
                        }
                    )

    print(
        f"Loaded {len(repo_by_id)} repos, {len(replaces_pairs)} declared replacement relationships."
    )

    cases: list[dict[str, Any]] = []
    case_idx = 1

    # =========================================================================
    # Type 1: Out-of-domain / Non-existent / Hallucination-trap (目标：弃权拒识)
    # 期望系统能够识别出根本不存在该项目，置信度应极低或 abstained=True
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
    ]
    languages = ["rust", "golang", "python", "typescript", "c++", "zig", "swift"]

    print("Generating Type 1: Out-of-domain / No-result trap queries...")
    # 1. 物理不可能/纯幻想技术
    for text in fictional_technologies:
        for _ in range(5):
            lang = random.choice(languages)
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

    # 2. 虚构库名
    for prefix in fictional_framework_prefixes:
        for lang in languages[:4]:
            cases.append(
                {
                    "id": f"neg-no-result-{case_idx:04d}",
                    "negative_type": "out_of_domain_no_result",
                    "query": f"{prefix}-{lang} framework for cloud orchestration",
                    "expected_repo_id": None,
                    "forbidden_top_repo_ids": [],
                    "expected_behavior": "abstain",
                    "tags": ["negative", "out-of-domain", "no-result", f"language-{lang}"],
                }
            )
            case_idx += 1

    # =========================================================================
    # Type 2: Alternative Contrastive (替代品测试：严禁推原主)
    # 用户搜 "X alternative"，第一名如果还是 X 本身，则判定测试失败！
    # =========================================================================
    print("Generating Type 2: Alternative contrastive queries (Forbidden target)...")
    famous_targets = [
        ("redis", ["redis/redis", "antirez/redis"]),
        ("firebase", ["firebase/firebase-js-sdk", "firebase/firebase-ios-sdk"]),
        ("notion", ["notion/notion"]),
        ("docker", ["docker/docker", "moby/moby"]),
        ("postman", ["postmanlabs/postman-app-support"]),
        ("elasticsearch", ["elastic/elasticsearch"]),
        ("nginx", ["nginx/nginx"]),
        ("kafka", ["apache/kafka"]),
        ("jira", ["atlassian/jira"]),
        ("airdrop", []),
        ("slack", ["tinyspeck/slack"]),
        ("trello", []),
        ("github", []),
        ("s3", ["aws/aws-sdk-js"]),
    ]

    for target, forbidden_rids in famous_targets:
        queries = [
            f"open source alternative to {target}",
            f"tools like {target} for self hosting",
            f"lightweight replacement for {target}",
            f"{target} alternative open source",
            f"modern drop-in replacement for {target}",
        ]
        for q in queries:
            cases.append(
                {
                    "id": f"neg-alternative-{case_idx:04d}",
                    "negative_type": "alternative_contrastive",
                    "query": q,
                    "target_project": target,
                    "forbidden_top_repo_ids": forbidden_rids,
                    "expected_behavior": "suppress_forbidden_target",
                    "tags": ["negative", "alternative", "contrastive", f"target-{target}"],
                }
            )
            case_idx += 1

    # 结合从 profile 里提取出来的真实替代品
    for pair in random.sample(replaces_pairs, min(200, len(replaces_pairs))):
        target = pair["target_name"]
        expected_alternative = pair["candidate_repo_id"]
        # 寻找原版库名在库里的所有 repo_id 作为禁用列表
        matching_targets = [
            rid for name, rids in repo_by_name.items() if name == target for rid in rids
        ]

        cases.append(
            {
                "id": f"neg-alternative-{case_idx:04d}",
                "negative_type": "alternative_contrastive",
                "query": f"open source alternative to {target}",
                "target_project": target,
                "expected_alternative_repo_id": expected_alternative,
                "forbidden_top_repo_ids": matching_targets,
                "expected_behavior": "suppress_forbidden_target",
                "tags": ["negative", "alternative", f"target-{target}"],
            }
        )
        case_idx += 1

    # =========================================================================
    # Type 3: Cross-Language Conflict (跨语言冲突排除)
    # 搜 "Rust web framework like express"，第一名绝不能是 Node 的 expressjs/express！
    # =========================================================================
    print("Generating Type 3: Cross-language conflict queries...")
    cross_lang_benchmarks = [
        ("Rust", "express", ["expressjs/express"]),
        ("Go", "flask", ["pallets/flask"]),
        ("Python", "tokio", ["tokio-rs/tokio"]),
        ("TypeScript", "gin", ["gin-gonic/gin"]),
        ("Rust", "django", ["django/django"]),
        ("C++", "numpy", ["numpy/numpy"]),
        ("Go", "celery", ["celery/celery"]),
        ("Python", "spring boot", ["spring-projects/spring-boot"]),
        ("Rust", "redis", ["redis/redis"]),
        ("Go", "pydantic", ["pydantic/pydantic"]),
        ("TypeScript", "fastapi", ["fastapi/fastapi", "tiangolo/fastapi"]),
        ("Rust", "puppeteer", ["puppeteer/puppeteer"]),
    ]

    for req_lang, ref_tool, forbidden_rids in cross_lang_benchmarks:
        templates = [
            f"{req_lang} web framework like {ref_tool}",
            f"{req_lang} library similar to {ref_tool}",
            f"port of {ref_tool} written in {req_lang}",
            f"{req_lang} implementation of {ref_tool}",
            f"lightweight {req_lang} alternative to {ref_tool}",
        ]
        for q in templates:
            cases.append(
                {
                    "id": f"neg-cross-lang-{case_idx:04d}",
                    "negative_type": "cross_language_conflict",
                    "query": q,
                    "requested_language": req_lang.lower(),
                    "forbidden_top_repo_ids": forbidden_rids,
                    "expected_behavior": "match_language_and_suppress_forbidden",
                    "tags": ["negative", "cross-language", f"target-{ref_tool.replace(' ', '-')}"],
                }
            )
            case_idx += 1

    # =========================================================================
    # Type 4: Exact Name Collision vs Distractor (精确名对抗诱饵库)
    # 用户搜 "vue"，不能把 "vue-router"、"vue-loader" 或 "awesome-vue" 排在 "vuejs/vue" 前面！
    # =========================================================================
    print("Generating Type 4: Exact name collision vs distractor queries...")
    canonical_targets = [
        ("vue", "vuejs/vue", ["vuejs/vue-router", "vuejs/vuex", "vuejs/vue-cli"]),
        (
            "react",
            "react/react",
            ["facebook/react-native", "reactjs/react-router", "enaqx/awesome-react"],
        ),
        ("gin", "gin-gonic/gin", ["gin-gonic/contrib", "gin-gonic/website"]),
        ("flask", "pallets/flask", ["miguelgrinberg/Flask-SocketIO", "pallets/flask-sqlalchemy"]),
        ("ansible", "ansible/ansible", ["ansible/ansible-lint", "ansible/ansible-runner"]),
        ("webpack", "webpack/webpack", ["webpack/webpack-cli", "webpack/webpack-dev-server"]),
        (
            "electron",
            "electron/electron",
            ["electron/electron-quick-start", "electron-userland/electron-builder"],
        ),
        ("flutter", "flutter/flutter", ["flutter/samples", "flutter/plugins"]),
        (
            "django",
            "django/django",
            ["encode/django-rest-framework", "django-debug-toolbar/django-debug-toolbar"],
        ),
    ]

    for name, canonical_rid, distractors in canonical_targets:
        cases.append(
            {
                "id": f"neg-distractor-{case_idx:04d}",
                "negative_type": "name_collision_distractor",
                "query": name,
                "expected_canonical_repo_id": canonical_rid,
                "forbidden_top_repo_ids": distractors,
                "expected_behavior": "canonical_beats_distractors",
                "tags": ["negative", "distractor-defense", f"canonical-{name}"],
            }
        )
        case_idx += 1
        # 加上小写精确搜
        cases.append(
            {
                "id": f"neg-distractor-{case_idx:04d}",
                "negative_type": "name_collision_distractor",
                "query": f"github {name}",
                "expected_canonical_repo_id": canonical_rid,
                "forbidden_top_repo_ids": distractors,
                "expected_behavior": "canonical_beats_distractors",
                "tags": ["negative", "distractor-defense", f"canonical-{name}"],
            }
        )
        case_idx += 1

    # 随机打乱以保证评测独立性
    random.seed(42)
    random.shuffle(cases)

    # 截取到目标大小 (如果超过)
    final_cases = cases[:total_target] if len(cases) > total_target else cases

    # 统计类型分布
    type_counts: dict[str, int] = {}
    for c in final_cases:
        t = c["negative_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    dataset = {
        "benchmark_name": "xists-negative-evaluation-1k",
        "description": "Adversarial negative evaluation cases for abstention, contrastive alternative routing, cross-language conflict suppression, and exact-name distractor defense.",
        "case_count": len(final_cases),
        "type_breakdown": type_counts,
        "cases": final_cases,
    }

    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSuccessfully generated {len(final_cases)} negative benchmark cases to {output_path}")
    print("Type breakdown:")
    for k, v in type_counts.items():
        print(f"  - {k}: {v} cases")
    return dataset


if __name__ == "__main__":
    build_negative_eval_dataset()
