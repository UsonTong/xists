import tomllib

import xists


def test_package_version_matches_project_metadata():
    with open("pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/xists/__init__.py"
    assert xists.__version__ == "0.7.0"
