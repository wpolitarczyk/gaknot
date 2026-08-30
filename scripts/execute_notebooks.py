"""Execute the project's tutorial notebooks in clean SageMath kernels.

The checker deliberately does not write the executed notebooks back to disk.
Committed notebooks therefore stay free of execution counts and output while
CI and local development can still verify that every narrative example runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_DIRECTORY = REPOSITORY_ROOT / "notebooks"


def notebook_paths(targets: list[str]) -> list[Path]:
    """Resolve file or directory arguments into an ordered notebook list."""

    requested = (
        [Path(target) for target in targets]
        or [DEFAULT_NOTEBOOK_DIRECTORY]
    )
    notebooks: list[Path] = []

    for requested_path in requested:
        path = (
            requested_path
            if requested_path.is_absolute()
            else REPOSITORY_ROOT / requested_path
        )
        if path.is_dir():
            notebooks.extend(sorted(path.glob("*.ipynb")))
        elif path.suffix == ".ipynb" and path.is_file():
            notebooks.append(path)
        else:
            raise FileNotFoundError(f"notebook target does not exist: {path}")

    if not notebooks:
        raise FileNotFoundError(
            "no .ipynb files were found in the requested targets"
        )

    # Preserve the first occurrence if a file is named both directly and via a
    # directory. Dict insertion order gives deterministic execution order.
    return list(dict.fromkeys(path.resolve() for path in notebooks))


def execute_notebook(path: Path, timeout: int) -> None:
    """Execute one notebook from the repository root without saving outputs."""

    with path.open(encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="sagemath",
        allow_errors=False,
        resources={"metadata": {"path": str(REPOSITORY_ROOT)}},
    )
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute tutorial notebooks in fresh SageMath kernels."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Notebook files or directories (default: notebooks/).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds (default: 600).",
    )
    arguments = parser.parse_args()

    paths = notebook_paths(arguments.targets)
    for index, path in enumerate(paths, start=1):
        try:
            display_path = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            display_path = path
        print(f"[{index}/{len(paths)}] Executing {display_path}", flush=True)
        execute_notebook(path, arguments.timeout)
        print(f"      Passed {display_path}", flush=True)


if __name__ == "__main__":
    main()
