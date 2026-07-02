import tomllib

import xists


def test_package_version_matches_project_metadata():
    with open("pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    assert xists.__version__ == pyproject["project"]["version"]
