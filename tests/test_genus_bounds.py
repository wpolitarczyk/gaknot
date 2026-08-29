r"""Tests for the supported Gilmer four-genus obstruction.

The low-dimensional linking-form tests are independent finite-field checks.
The final integration test reconstructs the knot from Theorem 1.1 of
Marchwicka--Politarczyk and verifies both primary computations from Lemma 3.1,
including the projective and isotropic-line counts.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import Integer, QQ

from gaknot import (
    GeneralizedAlgebraicKnot,
    GenusObstructionResult,
    PrimeDiagonalLinkingForm,
    gilmer_genus_obstruction,
)


def _published_genus_two_knot():
    """Return the eight-summand knot from the paper's Theorem 1.1."""
    return GeneralizedAlgebraicKnot([
        (1, [(2, 17), (2, 83)]),
        (-1, [(2, 11), (2, 83)]),
        (1, [(2, 83)]),
        (-1, [(2, 13), (2, 83)]),
        (1, [(2, 11), (2, 103)]),
        (-1, [(2, 103)]),
        (1, [(2, 13), (2, 103)]),
        (-1, [(2, 17), (2, 103)]),
    ])


# ---------------------------------------------------------------------------
# Prime diagonal linking forms
# ---------------------------------------------------------------------------

def test_prime_diagonal_linking_form_pairing_and_isotropy():
    form = PrimeDiagonalLinkingForm([5, 5], [1, -1])

    assert form.orders == (5, 5)
    assert form.coefficients == (1, -1)
    assert form.rank == 2
    assert form.primary_primes == (5,)
    assert form.primary_indices(5) == (0, 1)
    assert form.pairing([1, 0], [1, 0]) == QQ(1) / 5
    assert form.pairing([1, 1], [1, 1]) == 0
    assert not form.is_isotropic([1, 0])
    assert form.is_isotropic([1, 1])
    # Coordinates are elements of cyclic groups, so normalization must not
    # change the pairing represented by negative or oversized integers.
    assert form.pairing([-4, 6], [1, 1]) == 0
    assert repr(form) == (
        "PrimeDiagonalLinkingForm(orders=(5, 5), coefficients=(1, -1))"
    )


def test_prime_diagonal_linking_form_from_signed_cable_sum():
    knot = (
        GeneralizedAlgebraicKnot.torus_knot(2, 5)
        - GeneralizedAlgebraicKnot.torus_knot(2, 7)
    )

    form = PrimeDiagonalLinkingForm.from_knot(knot)

    # A positive cable has distinguished pairing -1/q; mirroring reverses it.
    assert form.orders == (5, 7)
    assert form.coefficients == (-1, 1)
    assert form.primary_indices(5) == (0,)
    assert form.primary_indices(7) == (1,)


def test_prime_diagonal_linking_form_metabolizer():
    hyperbolic = PrimeDiagonalLinkingForm([5, 5], [1, -1])

    assert hyperbolic.is_metabolizer([[1, 1]])
    assert hyperbolic.is_metabolizer([[1, -1]])
    assert not hyperbolic.is_metabolizer([[1, 0]])
    assert not hyperbolic.is_metabolizer([])


def test_metabolizer_generators_may_mix_primary_parts():
    form = PrimeDiagonalLinkingForm(
        [3, 3, 5, 5],
        [1, -1, 1, -1],
    )

    # This one element has order 15.  Its projections generate the isotropic
    # line in each two-dimensional primary part, so the subgroup has order 15
    # and is a metabolizer of the group of order 15^2.
    assert form.is_metabolizer([[1, 1, 1, 1]])
    assert not form.is_metabolizer([[1, 1, 1, 0]])


def test_projective_isotropic_elements_are_unique_canonical_lines():
    form = PrimeDiagonalLinkingForm([3, 5, 5], [1, 1, -1])

    representatives = list(form.projective_isotropic_elements(5))

    # x^2-y^2=0 has the two projective solutions [1:1] and [1:-1].  The
    # unrelated 3-primary coordinate stays zero.
    assert representatives == [(0, 1, 1), (0, 1, 4)]
    assert all(form.is_isotropic(vector) for vector in representatives)


@pytest.mark.parametrize(
    "orders, coefficients, error_type, match",
    [
        ("5", [1], TypeError, "orders must be"),
        ([5], "1", TypeError, "coefficients must be"),
        ([], [], ValueError, "at least one coordinate"),
        ([5, 7], [1], ValueError, "equal length"),
        ([4], [1], ValueError, "must be prime"),
        ([5], [5], ValueError, "not a unit"),
        ([True], [1], TypeError, "Order at index 0"),
        ([5], [QQ(1) / 5], TypeError, "Coefficient at index 0"),
    ],
)
def test_prime_diagonal_linking_form_validation(
    orders,
    coefficients,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        PrimeDiagonalLinkingForm(orders, coefficients)


@pytest.mark.parametrize(
    "element, error_type, match",
    [
        (1, TypeError, "list or tuple"),
        ([1], ValueError, "Expected 2"),
        ([1, QQ(1) / 5], TypeError, "coordinate at index 1"),
    ],
)
def test_prime_diagonal_linking_form_element_validation(
    element,
    error_type,
    match,
):
    form = PrimeDiagonalLinkingForm([5, 5], [1, -1])

    with pytest.raises(error_type, match=match):
        form.is_isotropic(element)


# ---------------------------------------------------------------------------
# Structured genus-obstruction results
# ---------------------------------------------------------------------------

def test_slice_connected_sum_is_reported_as_inconclusive():
    cinquefoil = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    slice_knot = cinquefoil - cinquefoil

    result = slice_knot.gilmer_genus_obstruction(0)
    primary = result.primary_checks[0]

    assert isinstance(result, GenusObstructionResult)
    assert result.tested_genus == 0
    assert result.classical_signature == 0
    assert not result.certified
    assert result.lower_bound is None
    assert result.successful_primes == ()
    assert primary.prime == 5
    assert primary.eligible
    assert primary.projective_vectors_in_search_space == 6
    # The first isotropic line already has cancelling Casson--Gordon
    # signatures for all multiples, so it is returned as the obstruction to
    # this sufficient search.
    assert primary.isotropic_lines_examined == 1
    assert primary.violating_lines == 0
    assert primary.unresolved_isotropic_element == (1, 1)


def test_anisotropic_primary_part_certifies_no_genus_zero_surface():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    result = gilmer_genus_obstruction(knot, 0)
    primary = result.primary_checks[0]

    # A nonzero one-dimensional primary form has no nonzero isotropic element.
    # It therefore cannot be the metabolic beta_2 forced by a genus-zero
    # decomposition.  This is a valid vacuous instance of the criterion.
    assert result.certified
    assert result.lower_bound == 1
    assert result.successful_primes == (5,)
    assert primary.projective_vectors_in_search_space == 1
    assert primary.isotropic_lines_examined == 0
    assert primary.violating_lines == 0
    assert primary.sample_witness is None


@pytest.mark.parametrize("genus", [True, False, QQ(1) / 2, 1.0, "1", None])
def test_gilmer_genus_obstruction_rejects_noninteger_genus(genus):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(TypeError, match="tested genus must be an integer"):
        knot.gilmer_genus_obstruction(genus)


def test_gilmer_genus_obstruction_rejects_negative_genus():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(ValueError, match="must be nonnegative"):
        knot.gilmer_genus_obstruction(-1)


def test_genus_obstruction_results_are_immutable():
    result = GeneralizedAlgebraicKnot.torus_knot(2, 5).gilmer_genus_obstruction(0)

    with pytest.raises(FrozenInstanceError):
        result.certified = False
    with pytest.raises(FrozenInstanceError):
        result.primary_checks[0].certified = False


# ---------------------------------------------------------------------------
# Published genus-two computation
# ---------------------------------------------------------------------------

def test_published_genus_two_knot_satisfies_primary_gilmer_search():
    knot = _published_genus_two_knot()

    result = knot.gilmer_genus_obstruction(1)

    assert result.classical_signature == 0
    assert result.certified
    assert result.lower_bound == 2
    assert result.successful_primes == (83, 103)
    assert result.linking_form.orders == (83,) * 4 + (103,) * 4

    for primary in result.primary_checks:
        prime = int(primary.prime)
        # The number of lines in projective 3-space over F_p is
        # p^3+p^2+p+1.  The split four-dimensional form has (p+1)^2
        # isotropic lines, and Lemma 3.1 requires a violation on every one.
        assert primary.projective_vectors_in_search_space == (
            prime ** 3 + prime ** 2 + prime + 1
        )
        assert primary.isotropic_lines_examined == (prime + 1) ** 2
        assert primary.violating_lines == primary.isotropic_lines_examined
        assert primary.certified
        assert primary.unresolved_isotropic_element is None

        witness = primary.sample_witness
        invariant = knot.casson_gordon(witness.character_parameters)
        assert invariant.sigma == witness.sigma
        assert invariant.eta == witness.eta
        assert witness.left_hand_side == abs(
            witness.sigma + witness.classical_signature
        )
        assert witness.left_hand_side > witness.bound

    # The bound is genuinely genus-dependent.  For g=2 each four-generator
    # primary part can fit into the rank-4 beta_1 allowed by the theorem, so
    # this particular sufficient search makes no assertion about genus >2.
    genus_two_test = knot.gilmer_genus_obstruction(2)
    assert not genus_two_test.certified
    assert genus_two_test.lower_bound is None
    assert all(not check.eligible for check in genus_two_test.primary_checks)
