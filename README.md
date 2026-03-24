# Generalized Algebraic Knot Invariants

[![CI Tests](https://github.com/wpolitarczyk/signature-function/actions/workflows/test-notebooks.yml/badge.svg)](https://github.com/wpolitarczyk/signature-function/actions/workflows/test-notebooks.yml)
[![pipeline status](https://gitlab.mimuw.edu.pl/wpolitarczyk/signature-function/badges/master/pipeline.svg)](https://gitlab.mimuw.edu.pl/wpolitarczyk/signature-function/-/commits/master)

This project allows calculating signature invariants for generalized algebraic knots (`GA-knots`).
Generalized algebraic knots are knots constructed as connected sums of positive iterated torus knots and their concordance inverses.
It was created as part of the proof for the main lemma from the paper **"On the slice genus of generalized algebraic knots"** (Maria Marchwicka and Wojciech Politarczyk) ([arXiv:2107.11299](https://arxiv.org/abs/2107.11299)).

## Project Structure

The project is organized as follows:

* **`gaknot/`**: Contains the source code (SageMath and Python files). This is the main package directory.
* **`test/`**: Contains Jupyter notebooks for testing and reproducing calculations using `ipytest`:
  * `lemma.ipynb`: Core calculations for the proof of Lemma 3.2.
  * `LT-signature-test.ipynb`: Tests for the core Levine-Tristram signature module.
  * `gaknot-test.ipynb`: Tests and usage examples for the `GeneralizedAlgebraicKnot` class.
  * `H1_branched_cover-test.ipynb`: Tests for the branched cover homology module.

## Requirements

* **SageMath**: This project relies on the SageMath kernel. Ensure you have a working installation of SageMath (e.g., via Conda or a native install).
* **ipytest**: Required for running the test suite within notebooks.

## Installation and Usage
The `gaknot` package is written using SageMath's `.sage` format. Before these modules can be imported into a standard Python environment or another Sage script, they must be **preparsed** into `.py` files.

### Recommended Import Method: `import_sage`
The project includes a utility function `import_sage` in `gaknot/utility.py` that automatically handles the preparsing and loading of `.sage` modules. This is the recommended way to import any module from the `gaknot` package.

```python
from gaknot.utility import import_sage

# This will preparse gaknot.sage into gaknot.py and import it
gaknot_mod = import_sage('gaknot', package='gaknot')
GeneralizedAlgebraicKnot = gaknot_mod.GeneralizedAlgebraicKnot
```

### Manual Installation
You can also use the following standard methods, provided you have manually preparsed the files (e.g., using `sage --preparse <file>.sage`):

**Option 1: Environment Variable**
Add the directory containing the `gaknot` folder to the `SAGE_PATH` environment variable.
```bash
export SAGE_PATH="/absolute/path/to/directory_containing_gaknot:$SAGE_PATH"
```

**Option 2: Manual Path Addition**
Append the directory to `sys.path` within your script:
```python
import sys
sys.path.append("/path/to/directory_containing_gaknot")
```

## Usage

### 1. Working with Generalized Algebraic Knots
The easiest way to work with the package is using the `GeneralizedAlgebraicKnot` class.

```python
from gaknot.utility import import_sage
gaknot_mod = import_sage('gaknot', package='gaknot')
GeneralizedAlgebraicKnot = gaknot_mod.GeneralizedAlgebraicKnot

# Define T(2,3) and T(3,4)
knot1 = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
knot2 = GeneralizedAlgebraicKnot([(1, [(3, 4)])])

# Define an iterated torus knot: (6,5)-cable of T(2,3)
knot_iter = GeneralizedAlgebraicKnot([(1, [(2, 3), (6, 5)])])

# Operations: Connected sum (+) and Concordance inverse (-)
sum_knot = knot1 + knot2
inverse_knot = -knot1

# Human-readable string representation
print(sum_knot)
# Output: T(2,3) # T(3,4)

print(knot_iter)
# Output: T(2,3; 6,5)
```

**Computing the Signature:**
You can extract the Levine-Tristram signature directly from the knot object.

```python
# Create the algebraically slice knot T(2,3) # -T(2,3)
slice_knot = knot1 + (-knot1)

# Compute the signature function
sig_func = slice_knot.signature()

# Verify that the signature function is zero
print(sig_func.is_zero_everywhere())
# Output: True
```

We can also test it on Litherland's example. This is a nontrivial generalized algebraic knot whose signature function is trivial.

```python
# Define the knot T(2,3;5,2) # T(3,2) # T(5,3) # -T(6,5)
desc = [
    (1, [(2,3), (5,2)]),
    (1, [(3,2)]),
    (1, [(5,3)]),
    (-1, [(6,5)])
]

alg_slice_knot = GeneralizedAlgebraicKnot(desc)

# compute the LT_signature
sig_func = alg_slice_knot.signature()

# Verify if it evaluates to 0 (since it's an algebraically slice knot)
print(f"Is the signature zero everywhere? {sig_func.is_zero_everywhere()}")
# Output: Is the signature zero everywhere? True
```

### 2. Branched Cover Homology
The `BranchedCoverHomology` class (available after preparsing `H1_branched_cover.sage`) computes the first homology group $H_1(\Sigma_N(K))$. It preserves the satellite structure decomposition.

```python
h1_mod = import_sage('H1_branched_cover', package='gaknot')
BranchedCoverHomology = h1_mod.BranchedCoverHomology

# Compute H_1 of the 2-fold branched cover of the trefoil
h1 = BranchedCoverHomology(knot1, 2)
print(h1) # Output: (Z/3Z)[T(2,3)]

# Compute H_1 of a connected sum
knot_sum = knot1 + (-knot_iter)
h1_sum = BranchedCoverHomology(knot_sum, 2)
print(h1_sum)
# Output: (Z/3Z)[T(2,3)] ⊕ (Z/5Z)[-T(2,3; 6,5)]

# Access invariant factors
print(h1_sum.invariant_factors) # Output: [3, 5]
```

### 3. The `LT_signature` module
If you need to bypass the class wrapper, you can compute signatures directly using the functional modules.

```python
lt_mod = import_sage('LT_signature', package='gaknot')
LT_signature_torus_knot = lt_mod.LT_signature_torus_knot
LT_signature_iterated_torus_knot = lt_mod.LT_signature_iterated_torus_knot

# Calculate signature for T(2,3)
sig = LT_signature_torus_knot(2, 3)

# Calculate signature for an iterated torus knot T(2,3; 6,5)
sig_iter = LT_signature_iterated_torus_knot([(2, 3), (6, 5)])
```

### 4. Visualization
The `SignatureFunction` objects can be evaluated and plotted.

```python
sig_mod = import_sage('signature', package='gaknot')
SignaturePloter = sig_mod.SignaturePloter

# Evaluate at a specific theta
print(sig(0.5)) # Output: -2

# Plot the step function
SignaturePloter.plot(sig, title="Signature of T(2,3)")

# Plot the signature of a connected sum
sig_sum = sig + sig_iter
SignaturePloter.plot(sig_sum, title="Signature of T(2,3) # T(2,3; 6,5)")
```

## Development and Testing

### Running Tests
The project uses `ipytest` within Jupyter notebooks for testing. To run tests, open the relevant `-test.ipynb` notebook and execute the cells. Each test cell typically includes a `%preparse` magic command to ensure the latest source code is used.


## Documentation
For a more detailed description of the classes, validation rules, and internal logic, please refer to the docstrings within the `gaknot` package files or explore the `test/` directory.
