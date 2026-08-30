# Executable tutorials

This directory contains a guided introduction to the public `gaknot` API.
The notebooks are numbered because later examples build on conventions and
objects introduced earlier.

| Notebook | Main topics |
| --- | --- |
| [`00_getting_started.ipynb`](00_getting_started.ipynb) | Environment checks, the GA-knot data model, constructors, connected sums, and basic invariants |
| [`01_signatures.ipynb`](01_signatures.ipynb) | Sparse jump data, midpoint values, periodicity, signature-function algebra, and plotting |
| [`02_branched_covers_and_characters.ipynb`](02_branched_covers_and_characters.ipynb) | Branched-cover homology, structural coordinates, characters, deck orbits, and twisted Alexander polynomials |
| [`03_casson_gordon_and_genus_bounds.ipynb`](03_casson_gordon_and_genus_bounds.ipynb) | Casson--Gordon summands, nullity, linking forms, isotropic lines, and Gilmer genus obstructions |
| [`04_metabelian_twisted_signatures.ipynb`](04_metabelian_twisted_signatures.ipynb) | Yanagida local models, Theorem 4.19, normalized twisted signatures, and explicit coverage gaps |

## Starting Jupyter with SageMath

Create and activate the environment described by the repository, build the
Sage sources, and start Jupyter from the repository root:

```bash
conda env create --file environment.yml   # needed only once
conda activate sage_env
make build
sage -n jupyter notebooks/
```

If the environment already exists, update it with
`conda env update --file environment.yml` instead of recreating it. Each
notebook records the `SageMath 10.7` kernel. If Jupyter asks for a kernel,
select **SageMath** rather than a plain Python kernel: the examples import
exact rings, matrices, and roots of unity from Sage.

The repository does not have to be installed before opening the notebooks.
Their first code cell adds `src/` to `sys.path`, so they run directly from a
checkout after `make build`. An editable installation produced by
`make install` works as well.

## Reproducible execution

The committed notebooks intentionally contain no execution counts or saved
outputs. This keeps reviews focused on source changes and prevents stale
results from looking authoritative. `make prep_commit` removes outputs,
execution counts, and transient execution metadata from every notebook in the
repository. To execute every notebook in a fresh Sage kernel without modifying
the files, run:

```bash
make notebooks
```

To check one notebook while editing it, use:

```bash
make notebooks NOTEBOOK_FILE=notebooks/02_branched_covers_and_characters.ipynb
```

Notebook execution is a tutorial-level smoke test. The pytest suite remains
the authoritative correctness and regression test suite; run it with
`make test`.

## Reading the mathematical output

All character values and arguments of signature functions are represented
exactly in `QQ` whenever possible. Arguments such as `1/5` represent the point
`exp(2*pi*i/5)` on the unit circle. The notebooks distinguish three kinds of
output carefully:

- a proved invariant value;
- a sufficient obstruction (whose failure is only inconclusive); and
- a coverage-aware partial result with explicitly unresolved roots.

That distinction is particularly important in the final two notebooks. The
package raises `NotImplementedError` when the available theorems do not
determine an exceptional local contribution; the tutorials show how to
inspect that limitation instead of concealing it.
