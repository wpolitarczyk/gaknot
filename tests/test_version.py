"""Regression tests for the public and packaging version declarations.

The package is commonly imported directly from ``src/`` during Sage
development, so users need a public ``gaknot.__version__`` even when no wheel
metadata has been installed.  The release number is also declared in
``pyproject.toml`` for packaging.  These tests ensure that the two sources
remain exact semantic versions and cannot silently diverge during a release.
"""

import re
from pathlib import Path

from gaknot import __version__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _pyproject_version():
    """Read the project version without requiring a TOML parser on Python 3.9."""
    pyproject_text = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    project_block = pyproject_text.split("[project]", 1)[1].split("[", 1)[0]
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        project_block,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError("No project version found in pyproject.toml.")
    return match.group(1)


def test_public_version_is_semantic_version():
    """Expose a conventional three-component release number to callers."""
    assert SEMANTIC_VERSION.fullmatch(__version__)


def test_public_version_matches_packaging_metadata():
    """Prevent imports and built distributions from reporting different releases."""
    assert __version__ == _pyproject_version()


def test_current_release_is_version_one():
    """Pin the intentional first stable release until the next version bump."""
    assert __version__ == "1.0.0"
