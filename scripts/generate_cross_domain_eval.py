"""Build fixed, cross-domain retrieval development and holdout datasets.

The query catalogue is deliberately hand-curated.  This script only expands
the catalogue into the public evaluation schema and verifies that every named
target is present in every supplied corpus.  Search results never influence
the generated labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DOMAINS = ("ai-llm", "web", "devtools", "infra", "data")
CATEGORIES = ("exact-name", "functional", "ecosystem", "ambiguous", "no-result")
LANGUAGES = ("en", "zh")


def _project(
    repo_id: str,
    name: str,
    functional_en: str,
    functional_zh: str,
    ecosystem_en: str,
    ecosystem_zh: str,
) -> dict[str, str]:
    return {
        "repo_id": repo_id,
        "name": name,
        "functional_en": functional_en,
        "functional_zh": functional_zh,
        "ecosystem_en": ecosystem_en,
        "ecosystem_zh": ecosystem_zh,
    }


# Every target below was manually selected from the 2k and 10k corpus lists.
# The generator verifies that claim before it writes an evaluation dataset.
PROJECTS: dict[str, list[dict[str, str]]] = {
    "ai-llm": [
        _project("huggingface/transformers", "Hugging Face Transformers", "Python library for pretrained transformer models", "用于预训练 Transformer 模型的 Python 库", "transformer machine learning ecosystem", "Transformer 机器学习生态"),
        _project("ollama/ollama", "Ollama", "run open language models locally", "在本地运行开源大语言模型", "local large language model runtime", "本地大语言模型运行环境"),
        _project("langchain-ai/langchain", "LangChain", "build applications and agents with large language models", "构建大语言模型应用和智能体", "Python LLM application framework", "Python 大语言模型应用框架"),
        _project("run-llama/llama_index", "LlamaIndex", "connect private data to large language model applications", "将私有数据接入大语言模型应用", "retrieval augmented generation framework", "检索增强生成框架"),
        _project("vllm-project/vllm", "vLLM", "serve large language models with high throughput", "高吞吐部署大语言模型服务", "GPU language model inference server", "GPU 大语言模型推理服务"),
        _project("open-webui/open-webui", "Open WebUI", "self-hosted web interface for local language models", "用于本地大语言模型的自托管 Web 界面", "self-hosted LLM application interface", "自托管大语言模型应用界面"),
    ],
    "web": [
        _project("react/react", "React", "JavaScript library for component user interfaces", "用于组件化用户界面的 JavaScript 库", "web user interface library ecosystem", "Web 用户界面库生态"),
        _project("vuejs/core", "Vue", "progressive JavaScript framework for web user interfaces", "渐进式 JavaScript Web 用户界面框架", "frontend component framework ecosystem", "前端组件框架生态"),
        _project("vercel/next.js", "Next.js", "React framework for server rendered web applications", "用于服务端渲染 Web 应用的 React 框架", "React full stack web framework", "React 全栈 Web 框架"),
        _project("fastapi/fastapi", "FastAPI", "Python framework for asynchronous REST APIs", "用于异步 REST API 的 Python 框架", "Python web API framework ecosystem", "Python Web API 框架生态"),
        _project("django/django", "Django", "Python framework for database backed web applications", "用于数据库驱动 Web 应用的 Python 框架", "Python server side web framework", "Python 服务端 Web 框架"),
        _project("expressjs/express", "Express", "minimal Node.js web application framework", "轻量级 Node.js Web 应用框架", "Node.js web framework ecosystem", "Node.js Web 框架生态"),
    ],
    "devtools": [
        _project("microsoft/vscode", "Visual Studio Code", "extensible source code editor", "可扩展的源代码编辑器", "developer editor tooling ecosystem", "开发者编辑器工具生态"),
        _project("neovim/neovim", "Neovim", "extensible terminal based text editor for developers", "面向开发者的可扩展终端文本编辑器", "terminal editor ecosystem", "终端编辑器生态"),
        _project("git/git", "Git", "distributed version control system", "分布式版本控制系统", "source control tooling ecosystem", "源代码版本控制工具生态"),
        _project("rust-lang/rust", "Rust", "systems programming language and compiler", "系统编程语言及其编译器", "systems programming language ecosystem", "系统编程语言生态"),
        _project("vitejs/vite", "Vite", "fast frontend development server and build tool", "快速前端开发服务器和构建工具", "modern frontend build tooling", "现代前端构建工具"),
        _project("prettier/prettier", "Prettier", "opinionated code formatter", "固定风格的代码格式化工具", "JavaScript code quality tooling", "JavaScript 代码质量工具生态"),
    ],
    "infra": [
        _project("kubernetes/kubernetes", "Kubernetes", "container orchestration platform", "容器编排平台", "cloud native container platform ecosystem", "云原生容器平台生态"),
        _project("hashicorp/terraform", "Terraform", "infrastructure as code provisioning tool", "基础设施即代码资源编排工具", "infrastructure as code ecosystem", "基础设施即代码生态"),
        _project("prometheus/prometheus", "Prometheus", "monitoring system and time series database", "监控系统和时序数据库", "cloud native monitoring ecosystem", "云原生监控生态"),
        _project("moby/moby", "Moby", "container engine components and tooling", "容器引擎组件和工具", "container runtime ecosystem", "容器运行时生态"),
        _project("ansible/ansible", "Ansible", "agentless infrastructure automation tool", "无代理基础设施自动化工具", "configuration management ecosystem", "配置管理生态"),
        _project("grafana/grafana", "Grafana", "observability dashboards and visualization platform", "可观测性仪表盘和可视化平台", "observability visualization ecosystem", "可观测性可视化生态"),
    ],
    "data": [
        _project("postgres/postgres", "PostgreSQL", "open source relational database", "开源关系型数据库", "relational database ecosystem", "关系型数据库生态"),
        _project("redis/redis", "Redis", "in memory data store and cache", "内存数据存储和缓存", "in memory data platform ecosystem", "内存数据平台生态"),
        _project("apache/kafka", "Apache Kafka", "distributed event streaming platform", "分布式事件流平台", "event streaming ecosystem", "事件流生态"),
        _project("apache/spark", "Apache Spark", "distributed data processing engine", "分布式数据处理引擎", "distributed data processing ecosystem", "分布式数据处理生态"),
        _project("duckdb/duckdb", "DuckDB", "analytical SQL database for local data", "面向本地数据的分析型 SQL 数据库", "analytical database ecosystem", "分析型数据库生态"),
        _project("pandas-dev/pandas", "pandas", "Python library for tabular data analysis", "用于表格数据分析的 Python 库", "Python data analysis ecosystem", "Python 数据分析生态"),
    ],
}


AMBIGUOUS: dict[str, list[tuple[str, str, list[str], str]]] = {
    "ai-llm": [
        ("framework for LLM applications and agents", "langchain-ai/langchain", ["run-llama/llama_index"], "en"),
        ("面向私有数据的大语言模型应用框架", "run-llama/llama_index", ["langchain-ai/langchain"], "zh"),
    ],
    "web": [
        ("component based JavaScript user interface library", "react/react", ["vuejs/core"], "en"),
        ("现代前端组件框架", "vuejs/core", ["react/react"], "zh"),
    ],
    "devtools": [
        ("extensible editor for software development", "microsoft/vscode", ["neovim/neovim"], "en"),
        ("可扩展的开发者代码编辑器", "neovim/neovim", ["microsoft/vscode"], "zh"),
    ],
    "infra": [
        ("infrastructure automation and provisioning tool", "hashicorp/terraform", ["ansible/ansible"], "en"),
        ("基础设施自动化与配置工具", "ansible/ansible", ["hashicorp/terraform"], "zh"),
    ],
    "data": [
        ("open source relational SQL database", "postgres/postgres", ["MariaDB/server"], "en"),
        ("开源关系型 SQL 数据库", "MariaDB/server", ["postgres/postgres"], "zh"),
    ],
}


NO_RESULT = {
    "ai-llm": ("open source compiler for telepathic neural weather models", "用于心灵感应神经天气模型的开源编译器"),
    "web": ("framework for legally binding interplanetary browser contracts", "用于星际浏览器合同的专用开源框架"),
    "devtools": ("source editor for reversible quantum legal histories", "用于可逆量子法律历史的源代码编辑器"),
    "infra": ("orchestrator for autonomous lunar water treaties", "管理月球水资源条约的容器编排系统"),
    "data": ("database for certified dream accounting ledgers", "用于认证梦境账本的专用数据库"),
}


def _case(
    *,
    case_id: str,
    query: str,
    expected_repo_id: str,
    domain: str,
    category: str,
    language: str,
    acceptable: list[str] | None = None,
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": case_id,
        "query": query,
        "expected_repo_id": expected_repo_id,
        "tags": [f"domain-{domain}", f"category-{category}", f"language-{language}", "curated"],
    }
    if acceptable:
        case["acceptable"] = acceptable
    return case


def _split_cases(split: str) -> list[dict[str, Any]]:
    if split not in {"dev", "holdout"}:
        raise ValueError(f"Unsupported split: {split}")
    offset = 0 if split == "dev" else 3
    cases: list[dict[str, Any]] = []
    for domain in DOMAINS:
        projects = PROJECTS[domain]
        selected = projects[offset : offset + 3]
        for index, project in enumerate(selected):
            language = "zh" if index == 1 else "en"
            query = f"查找 {project['name']} 开源项目" if language == "zh" else project["repo_id"]
            cases.append(_case(
                case_id=f"{split}-{domain}-exact-{index + 1}",
                query=query,
                expected_repo_id=project["repo_id"],
                domain=domain,
                category="exact-name",
                language=language,
            ))
        for index, project in enumerate(selected):
            language = "zh" if index == 2 else "en"
            cases.append(_case(
                case_id=f"{split}-{domain}-functional-{index + 1}",
                query=project[f"functional_{language}"],
                expected_repo_id=project["repo_id"],
                domain=domain,
                category="functional",
                language=language,
            ))
        ecosystem_projects = projects[(offset + 1) % 6 : (offset + 3) % 6]
        if len(ecosystem_projects) < 2:
            ecosystem_projects = [projects[(offset + 1) % 6], projects[(offset + 2) % 6]]
        for index, project in enumerate(ecosystem_projects):
            language = "zh" if index else "en"
            cases.append(_case(
                case_id=f"{split}-{domain}-ecosystem-{index + 1}",
                query=project[f"ecosystem_{language}"],
                expected_repo_id=project["repo_id"],
                domain=domain,
                category="ecosystem",
                language=language,
            ))
        for index, (query, expected, acceptable, language) in enumerate(AMBIGUOUS[domain], start=1):
            cases.append(_case(
                case_id=f"{split}-{domain}-ambiguous-{index}",
                query=query if split == "dev" else f"open source {query}" if language == "en" else f"开源 {query}",
                expected_repo_id=expected,
                acceptable=acceptable,
                domain=domain,
                category="ambiguous",
                language=language,
            ))
        for index, query in enumerate(NO_RESULT[domain], start=1):
            language = "en" if index == 1 else "zh"
            prefix = "dev" if split == "dev" else "holdout"
            cases.append(_case(
                case_id=f"{split}-{domain}-no-result-{index}",
                query=query if split == "dev" else f"open source {query}" if language == "en" else f"开源 {query}",
                expected_repo_id=f"xists/no-result-{prefix}-{domain}-{index}",
                domain=domain,
                category="no-result",
                language=language,
            ))
    return cases


def build_dataset(split: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_name": f"xists-cross-domain-{split}-v1",
        "cases": _split_cases(split),
    }


def validate_datasets(datasets: dict[str, dict[str, Any]], corpora: list[list[dict[str, Any]]]) -> None:
    all_queries: set[str] = set()
    all_ids: set[str] = set()
    corpus_ids = [{str(record.get("repo_id")) for record in records if record.get("repo_id")} for records in corpora]
    for split, dataset in datasets.items():
        coverage = {tag: set() for tag in ("domain", "category", "language")}
        if len(dataset["cases"]) != 60:
            raise ValueError(f"{split} must contain exactly 60 cases")
        for case in dataset["cases"]:
            if case["id"] in all_ids:
                raise ValueError(f"duplicate case id: {case['id']}")
            all_ids.add(case["id"])
            normalized_query = " ".join(case["query"].lower().split())
            if normalized_query in all_queries:
                raise ValueError(f"dev and holdout queries must be distinct: {case['query']}")
            all_queries.add(normalized_query)
            for tag in case["tags"]:
                prefix, _, value = tag.partition("-")
                if prefix in coverage:
                    coverage[prefix].add(value)
            expected = case["expected_repo_id"]
            if "category-no-result" in case["tags"]:
                if any(expected in ids for ids in corpus_ids):
                    raise ValueError(f"no-result sentinel is present in a corpus: {expected}")
            elif not all(expected in ids for ids in corpus_ids):
                raise ValueError(f"target is missing from one or more corpora: {expected}")
            for alternative in case.get("acceptable", []):
                if not all(alternative in ids for ids in corpus_ids):
                    raise ValueError(f"acceptable target is missing from one or more corpora: {alternative}")
        if coverage["domain"] != set(DOMAINS):
            raise ValueError(f"{split} does not cover every domain")
        if coverage["category"] != set(CATEGORIES):
            raise ValueError(f"{split} does not cover every category")
        if coverage["language"] != set(LANGUAGES):
            raise ValueError(f"{split} does not cover English and Chinese")


def _load_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"records file must contain a list: {path}")
    return [record for record in raw if isinstance(record, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixed cross-domain retrieval eval datasets")
    parser.add_argument("--records", action="append", type=Path, required=True, help="Corpus records to verify")
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    args = parser.parse_args()

    datasets = {"dev": build_dataset("dev"), "holdout": build_dataset("holdout")}
    validate_datasets(datasets, [_load_records(path) for path in args.records])
    for split, output in (("dev", args.dev_output), ("holdout", args.holdout_output)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(datasets[split], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
