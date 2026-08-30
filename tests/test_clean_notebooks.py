"""Tests for the pre-commit Jupyter notebook sanitizer.

These tests use a temporary notebook rather than any mathematical tutorial.
That makes it possible to verify removal of every generated field without
ever committing deliberately stale output to the real notebooks.
"""

import json

from scripts.clean_notebooks import clean_notebook


def test_clean_notebook_removes_execution_state_but_preserves_sources(tmp_path):
    """Code, Markdown, and durable metadata survive while results disappear."""

    notebook_path = tmp_path / "example.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"purpose": "explanation"},
                "source": "# Mathematical explanation",
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {
                    "ExecuteTime": {"start_time": "generated"},
                    "execution": {"iopub.status.busy": "generated"},
                    "trusted": True,
                    "tags": ["keep-this-tag"],
                },
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": "stale output\n",
                    }
                ],
                "source": "answer = 6 * 7",
            },
        ],
        "metadata": {
            "kernelspec": {"name": "sagemath"},
            "widgets": {"application/vnd.jupyter.widget-state+json": {}},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    assert clean_notebook(notebook_path)

    cleaned = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cell, code_cell = cleaned["cells"]
    assert markdown_cell == notebook["cells"][0]
    assert code_cell["source"] == "answer = 6 * 7"
    assert code_cell["outputs"] == []
    assert code_cell["execution_count"] is None
    assert code_cell["metadata"] == {"tags": ["keep-this-tag"]}
    assert cleaned["metadata"] == {"kernelspec": {"name": "sagemath"}}


def test_clean_notebook_does_not_rewrite_an_already_clean_file(tmp_path):
    """A clean notebook remains byte-for-byte stable across repeated cleanup."""

    notebook_path = tmp_path / "clean.ipynb"
    original = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "value = 1",
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        },
        separators=(",", ":"),
    )
    notebook_path.write_text(original, encoding="utf-8")

    assert not clean_notebook(notebook_path)
    assert notebook_path.read_text(encoding="utf-8") == original
