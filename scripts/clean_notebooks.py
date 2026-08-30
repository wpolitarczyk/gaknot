"""Remove generated execution state from every project notebook.

Notebook outputs are convenient while experimenting, but they quickly become
stale and make source reviews noisy. This script preserves cell sources and
durable descriptive metadata while removing code-cell results, execution
counts, execution-specific cell metadata, and cached widget state.

The implementation uses only Python's standard library so ``make
prep_commit`` remains a cleanup command that does not require SageMath or the
optional notebook dependencies to be importable by the active Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_CELL_METADATA = frozenset({
    "ExecuteTime",
    "execution",
    "trusted",
})


def notebook_paths(targets: list[str]) -> list[Path]:
    """Resolve files and directories to non-checkpoint notebooks."""

    requested = [Path(target) for target in targets] or [REPOSITORY_ROOT]
    notebooks: list[Path] = []

    for requested_path in requested:
        path = (
            requested_path
            if requested_path.is_absolute()
            else REPOSITORY_ROOT / requested_path
        )
        if path.is_dir():
            candidates = path.rglob("*.ipynb")
            notebooks.extend(
                candidate
                for candidate in candidates
                if ".ipynb_checkpoints" not in candidate.parts
                and ".git" not in candidate.parts
            )
        elif path.suffix == ".ipynb" and path.is_file():
            if ".ipynb_checkpoints" not in path.parts:
                notebooks.append(path)
        else:
            raise FileNotFoundError(f"notebook target does not exist: {path}")

    # A caller may name the same file directly and through a directory.
    return sorted(dict.fromkeys(path.resolve() for path in notebooks))


def clean_notebook(path: Path) -> bool:
    """Strip generated execution state and return whether the file changed."""

    with path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("outputs") != []:
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

        metadata = cell.setdefault("metadata", {})
        for key in TRANSIENT_CELL_METADATA:
            if key in metadata:
                del metadata[key]
                changed = True

    # Jupyter stores widget output state at notebook level rather than in one
    # cell. It is generated display data and must disappear with the outputs
    # that refer to it.
    notebook_metadata = notebook.setdefault("metadata", {})
    if "widgets" in notebook_metadata:
        del notebook_metadata["widgets"]
        changed = True

    if changed:
        serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        temporary_path = path.with_name(f".{path.name}.cleaning")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(path)

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove outputs and execution state from Jupyter notebooks."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Notebook files or directories (default: the whole repository).",
    )
    arguments = parser.parse_args()

    paths = notebook_paths(arguments.targets)
    changed_paths = [path for path in paths if clean_notebook(path)]

    for path in changed_paths:
        try:
            display_path = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            display_path = path
        print(f"Cleaned notebook execution state: {display_path}")

    print(
        f"Notebook cleanup complete: {len(changed_paths)} changed, "
        f"{len(paths) - len(changed_paths)} already clean."
    )


if __name__ == "__main__":
    main()
