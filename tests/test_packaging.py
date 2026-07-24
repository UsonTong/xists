import tomllib
from pathlib import Path

from xists import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_project_uses_package_version_as_the_single_source_of_truth():
    with (ROOT / "pyproject.toml").open("rb") as source:
        pyproject = tomllib.load(source)

    project = pyproject["project"]
    assert project["dynamic"] == ["version"]
    assert "version" not in project
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/xists/__init__.py"
    assert __version__ == "0.7.0"


def test_project_metadata_declares_license_and_public_urls():
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]

    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["urls"]["Repository"] == "https://github.com/UsonTong/xists"
    assert (ROOT / "LICENSE").is_file()


def test_sdist_build_whitelist_excludes_local_and_private_artifacts():
    with (ROOT / "pyproject.toml").open("rb") as source:
        build = tomllib.load(source)["tool"]["hatch"]["build"]

    assert "data" not in build["only-include"]
    assert ".claude" not in build["only-include"]
    assert "src/xists" in build["only-include"]
    assert "LICENSE" in build["only-include"]
