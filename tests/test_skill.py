from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "xists-project-search"


def test_xists_project_search_skill_declares_a_concise_mcp_workflow():
    content = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert content.startswith("---\nname: xists-project-search\n")
    assert "search_projects" in content
    assert "inspect_project" in content
    assert "index_stats" in content
    assert "at most five results" in content
    assert "strongest two candidates" in content
    assert "abstained: true" in content


def test_xists_project_search_skill_does_not_embed_personal_configuration():
    content = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "xists-local" not in content
    assert "EMBEDDING_API_KEY" not in content
    assert "/home/" not in content
    assert "index.json" not in content
    assert (SKILL / "agents" / "openai.yaml").is_file()
