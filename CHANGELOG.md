# Changelog

This file records user-visible changes to `gaknot`. The project follows
[Semantic Versioning](https://semver.org/): a major release identifies a
documented public API that users may rely on, not the completion of every
possible invariant for every generalized algebraic knot.

## 1.0.0 - 2026-09-01

Version 1.0.0 is the first stable release of the substantially rebuilt
SageMath package. It retains the project's original generalized-algebraic-knot
calculations while exposing their data, hypotheses, intermediate results, and
failure boundaries through documented public interfaces.

### Mathematical functionality

- Represent signed connected sums of iterated positive torus knots with
  defensive structural descriptions.
- Compute normalized Alexander polynomials and exact Levine--Tristram
  signature functions.
- Compute first homology of cyclic branched covers, structural homology
  elements, exact `Q/Z`-valued characters, and character deck orbits.
- Compute Casson--Gordon signatures and nullities for the supported
  `(2,q)`-cable families.
- Search Gilmer four-genus obstructions using primary linking forms,
  projective isotropic lines, explicit witnesses, and optional exhaustive
  logs.
- Construct Conway--Kim--Politarczyk metabelian representation matrices, Fox
  determinants, exterior and zero-surgery twisted Alexander representatives,
  cyclotomic root multiplicities, and cable levels.
- Construct Yanagida's exact local twisted Blanchfield matrices and
  coverage-aware signature-jump profiles.
- Assemble the supported divisible and nondivisible branches of the
  metabelian satellite signature formula, reporting unresolved local
  contributions explicitly.

### Documentation and verification

- Provide ordered executable SageMath notebooks ranging from introductory API
  examples to reconstructions of the Marchwicka--Politarczyk and
  Conway--Kim--Politarczyk calculations.
- Support optional deterministic text logs for theorem-sized computations.
- Provide configurable test verbosity and selection of one test file through
  the Makefile.
- Remove generated notebook output as part of `make prep_commit`.
- Verify the mathematical and API contracts with more than 770 regression
  tests.

### Explicit scope boundaries

- Casson--Gordon and Gilmer obstruction APIs currently implement the
  documented `(2,q)`-cable families rather than arbitrary GA-knots.
- Twisted Alexander formulas for negative torus knots are not yet implemented.
- Full metabelian signature functions are available only where the implemented
  local-pairing formulas cover every required root; partial results retain
  explicit coverage gaps instead of substituting conjectural values.
- The Conway--Kim--Politarczyk reconstruction does not yet enumerate general
  `Z_p`-invariant metabolizers or implement the complete Witt-class proof of
  their linear-independence theorem.

For installation, usage, citation guidance, and the mathematical bibliography,
see [`README.md`](README.md).
