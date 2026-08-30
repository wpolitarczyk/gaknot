r"""Tests for the supported Gilmer four-genus obstruction.

The low-dimensional linking-form tests are independent finite-field checks.
The final integration test reconstructs the knot from Theorem 1.1 of
Marchwicka--Politarczyk and verifies both primary computations from Lemma 3.1,
including the projective and isotropic-line counts.

The tests deliberately proceed in the same order as the obstruction itself:

* first verify the finite linking-form model, including exact pairings,
  isotropic elements, metabolizers, primary decompositions, and projective
  representatives;
* next inspect the structured outcomes of small obstruction searches, where
  both certification and inconclusiveness can be understood by hand; and
* finally reproduce the full published genus-two computation and audit a
  stored Casson--Gordon witness against an independent public-API call.

This separation matters because the final theorem-sized example alone would
not identify whether a failure came from linking-form coordinates, finite-field
enumeration, Casson--Gordon values, the Gilmer inequality, or result reporting.
Conversely, the small unit tests cannot establish that all pieces interact
correctly on the calculation for which the implementation was introduced.
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
    """Return the eight-summand knot from the paper's Theorem 1.1.

    The first four summands have outer order 83 and the final four have outer
    order 103.  Within each block the signs and companions are exactly those
    used in Marchwicka--Politarczyk.  Keeping this construction in one helper
    makes the integration test readable while leaving the full mathematical
    input visible in this test module.
    """
    return GeneralizedAlgebraicKnot([
        # The q=83 primary block: four distinguished Z/83 coordinates with
        # companion/sign choices matching the first half of the theorem knot.
        (1, [(2, 17), (2, 83)]),
        (-1, [(2, 11), (2, 83)]),
        (1, [(2, 83)]),
        (-1, [(2, 13), (2, 83)]),
        # The q=103 primary block: another four coordinates, ordered exactly
        # as in the paper and retained in positions 4 through 7 of the form.
        (1, [(2, 11), (2, 103)]),
        (-1, [(2, 103)]),
        (1, [(2, 13), (2, 103)]),
        (-1, [(2, 17), (2, 103)]),
    ])


# ---------------------------------------------------------------------------
# Prime diagonal linking forms
# ---------------------------------------------------------------------------

def test_prime_diagonal_linking_form_pairing_and_isotropy():
    """Check the basic algebra of a split rank-two form over ``Z/5``.

    The diagonal form ``<1/5> + <-1/5>`` is the smallest example containing
    both anisotropic vectors and nonzero isotropic vectors.  It therefore
    exercises representation metadata, exact ``Q/Z`` arithmetic, coordinate
    normalization, and the convenience isotropy predicate in one transparent
    calculation.
    """
    form = PrimeDiagonalLinkingForm([5, 5], [1, -1])

    # These properties describe the stored diagonal presentation.  Tuples are
    # expected so callers cannot mutate the form after it has been validated.
    assert form.orders == (5, 5)
    assert form.coefficients == (1, -1)
    assert form.rank == 2

    # Both coordinates belong to the same 5-primary block, and their original
    # positions are retained for translation back to knot components.
    assert form.primary_primes == (5,)
    assert form.primary_indices(5) == (0, 1)

    # On the first basis vector the self-pairing is 1/5.  On (1,1), the two
    # diagonal terms cancel in Q/Z: 1/5 - 1/5 = 0.
    assert form.pairing([1, 0], [1, 0]) == QQ(1) / 5
    assert form.pairing([1, 1], [1, 1]) == 0
    assert not form.is_isotropic([1, 0])
    assert form.is_isotropic([1, 1])

    # Coordinates are elements of cyclic groups, so normalization must not
    # change the pairing represented by negative or oversized integers.
    # Here (-4,6) reduces to (1,1) modulo 5.
    assert form.pairing([-4, 6], [1, 1]) == 0

    # ``repr`` is intended to preserve all constructor data and provide a
    # concise diagnostic when an obstruction result is inspected or fails.
    assert repr(form) == (
        "PrimeDiagonalLinkingForm(orders=(5, 5), coefficients=(1, -1))"
    )


def test_prime_diagonal_linking_form_from_signed_cable_sum():
    """Translate knot signs and outer orders into diagonal linking data.

    The example has two different primary orders so that the test also checks
    preservation of global component indices.  The positive ``T(2,5)``
    contributes ``-1/5``; the negative ``T(2,7)`` contributes ``+1/7``.
    """
    knot = (
        GeneralizedAlgebraicKnot.torus_knot(2, 5)
        - GeneralizedAlgebraicKnot.torus_knot(2, 7)
    )

    form = PrimeDiagonalLinkingForm.from_knot(knot)

    # A positive cable has distinguished pairing -1/q; mirroring reverses it.
    assert form.orders == (5, 7)
    assert form.coefficients == (-1, 1)

    # Primary decomposition must return indices in the original connected-sum
    # coordinate system, not local indices renumbered inside each block.
    assert form.primary_indices(5) == (0,)
    assert form.primary_indices(7) == (1,)


def test_prime_diagonal_linking_form_metabolizer():
    """Recognize exactly the isotropic half-dimensional lines in rank two.

    In the split form ``<1/5,-1/5>``, each of ``(1,1)`` and ``(1,-1)`` spans
    a one-dimensional isotropic subspace, hence a metabolizer.  The line
    generated by ``(1,0)`` has the correct dimension but is not isotropic,
    while the empty generating set is isotropic but has the wrong dimension.
    Together these failures ensure that both defining conditions are checked.
    """
    hyperbolic = PrimeDiagonalLinkingForm([5, 5], [1, -1])

    # Positive cases: rank one is half the ambient rank, and the restricted
    # Gram matrix vanishes.
    assert hyperbolic.is_metabolizer([[1, 1]])
    assert hyperbolic.is_metabolizer([[1, -1]])

    # Negative cases isolate failure of isotropy and failure of dimension.
    assert not hyperbolic.is_metabolizer([[1, 0]])
    assert not hyperbolic.is_metabolizer([])


def test_metabolizer_generators_may_mix_primary_parts():
    """Project a mixed-order generator into each primary vector space.

    A subgroup generator in a finite abelian group need not live in only one
    primary block.  The implementation must decompose its projections over
    ``F_3`` and ``F_5`` independently instead of trying to place the complete
    vector in a nonexistent common field.
    """
    form = PrimeDiagonalLinkingForm(
        [3, 3, 5, 5],
        [1, -1, 1, -1],
    )

    # This one element has order 15.  Its projections generate the isotropic
    # line in each two-dimensional primary part, so the subgroup has order 15
    # and is a metabolizer of the group of order 15^2.
    assert form.is_metabolizer([[1, 1, 1, 1]])

    # Removing the last coordinate leaves the 3-primary projection metabolic
    # but makes the 5-primary projection anisotropic, so the whole subgroup is
    # not a metabolizer.
    assert not form.is_metabolizer([[1, 1, 1, 0]])


def test_projective_isotropic_elements_are_unique_canonical_lines():
    """Enumerate each isotropic projective line exactly once.

    Only the two 5-primary coordinates participate.  Canonicalization scales
    each nonzero vector so its first nonzero local coordinate is one, removing
    the four other nonzero scalar multiples of the same line.  The leading
    3-primary coordinate must remain zero in the returned full-length vector.
    """
    form = PrimeDiagonalLinkingForm([3, 5, 5], [1, 1, -1])

    representatives = list(form.projective_isotropic_elements(5))

    # x^2-y^2=0 has the two projective solutions [1:1] and [1:-1].  The
    # unrelated 3-primary coordinate stays zero.
    assert representatives == [(0, 1, 1), (0, 1, 4)]

    # Recheck the generator's filtering criterion through the public pairing
    # API rather than trusting the enumeration's internal congruence test.
    assert all(form.is_isotropic(vector) for vector in representatives)


@pytest.mark.parametrize(
    "orders, coefficients, error_type, match",
    [
        # The two constructor arguments must be coordinate sequences, even for
        # a rank-one form; bare strings must not be treated as iterables here.
        ("5", [1], TypeError, "orders must be"),
        ([5], "1", TypeError, "coefficients must be"),
        # A linking form needs at least one coordinate, and both pieces of its
        # diagonal presentation must describe the same number of coordinates.
        ([], [], ValueError, "at least one coordinate"),
        ([5, 7], [1], ValueError, "equal length"),
        # This specialized class accepts prime cyclic summands and nonsingular
        # diagonal coefficients only.
        ([4], [1], ValueError, "must be prime"),
        ([5], [5], ValueError, "not a unit"),
        # Booleans and nonintegral exact rationals are rejected explicitly;
        # accepting either would obscure malformed algebraic input.
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
    """Reject malformed or degenerate diagonal presentations at construction.

    The table covers container shape, nonemptiness, equal lengths, prime-order
    cyclic factors, unit coefficients, and exact integer types.  Matching the
    relevant error fragment ensures the failure occurs for the intended
    reason rather than through an unrelated downstream exception.
    """
    with pytest.raises(error_type, match=match):
        PrimeDiagonalLinkingForm(orders, coefficients)


@pytest.mark.parametrize(
    "element, error_type, match",
    [
        # Elements use one explicit coordinate per diagonal cyclic summand.
        (1, TypeError, "list or tuple"),
        ([1], ValueError, "Expected 2"),
        # Coordinates must be exact integers before reduction modulo the
        # corresponding prime.
        ([1, QQ(1) / 5], TypeError, "coordinate at index 1"),
    ],
)
def test_prime_diagonal_linking_form_element_validation(
    element,
    error_type,
    match,
):
    """Validate element shape and coordinate types before pairing arithmetic.

    ``is_isotropic`` is used as the entry point because it delegates to the
    same element-normalization path as ``pairing``.  These cases therefore
    protect every public operation that accepts a linking-form vector.
    """
    form = PrimeDiagonalLinkingForm([5, 5], [1, -1])

    with pytest.raises(error_type, match=match):
        form.is_isotropic(element)


# ---------------------------------------------------------------------------
# Structured genus-obstruction results
# ---------------------------------------------------------------------------

def test_slice_connected_sum_is_reported_as_inconclusive():
    """Do not turn failure of this sufficient criterion into a genus claim.

    The structural knot ``T(2,5) # -T(2,5)`` is an ideal negative control: its
    two diagonal entries form a split block, and corresponding characters have
    cancelling Casson--Gordon signatures.  The search must stop at an
    unresolved isotropic line and report only that it failed to certify a
    positive four-genus.
    """
    cinquefoil = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    slice_knot = cinquefoil - cinquefoil

    result = slice_knot.gilmer_genus_obstruction(0)
    primary = result.primary_checks[0]

    # Check the top-level semantics first.  ``certified=False`` means
    # inconclusive, which is encoded by the absence of a numerical lower bound
    # and by an empty tuple of individually successful primes.
    assert isinstance(result, GenusObstructionResult)
    assert result.tested_genus == 0
    assert result.classical_signature == 0
    assert not result.certified
    assert result.lower_bound is None
    assert result.successful_primes == ()

    # The 5-primary block has two generators, exceeding 2g=0, so it is
    # eligible for the metabolizer argument.  Projective one-space over F_5
    # contains 5+1=6 lines before isotropy is imposed.
    assert primary.prime == 5
    assert primary.eligible
    assert primary.projective_vectors_in_search_space == 6
    # The first isotropic line already has cancelling Casson--Gordon
    # signatures for all multiples, so it is returned as the obstruction to
    # this sufficient search.
    assert primary.isotropic_lines_examined == 1
    assert primary.violating_lines == 0
    # Canonical projective normalization chooses (1,1) for the first isotropic
    # line on which no nonzero multiple violates Gilmer's inequality.
    assert primary.unresolved_isotropic_element == (1, 1)


def test_anisotropic_primary_part_certifies_no_genus_zero_surface():
    """Cover the valid vacuous-certification path for an anisotropic block.

    The linking form of a single ``T(2,5)`` is one-dimensional and
    nonsingular.  It contains a projective vector but no nonzero isotropic
    vector.  A hypothetical metabolic summand forced at genus zero would have
    to contain such a vector, so the absence itself supplies the obstruction.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    result = gilmer_genus_obstruction(knot, 0)
    primary = result.primary_checks[0]

    # A nonzero one-dimensional primary form has no nonzero isotropic element.
    # It therefore cannot be the metabolic beta_2 forced by a genus-zero
    # decomposition.  This is a valid vacuous instance of the criterion.
    assert result.certified
    assert result.lower_bound == 1
    assert result.successful_primes == (5,)

    # Projective zero-space has one line, represented by the nonzero vector
    # (1), but it is anisotropic and is therefore never yielded to the inner
    # Casson--Gordon search.
    assert primary.projective_vectors_in_search_space == 1
    assert primary.isotropic_lines_examined == 0
    assert primary.violating_lines == 0

    # Certification came from anisotropy rather than an inequality violation,
    # so there is no character witness to retain.
    assert primary.sample_witness is None


@pytest.mark.parametrize("genus", [True, False, QQ(1) / 2, 1.0, "1", None])
def test_gilmer_genus_obstruction_rejects_noninteger_genus(genus):
    """Require an exact integer value for the proposed surface genus.

    The cases distinguish booleans, an exact nonintegral Sage rational, a
    floating-point value, text, and a missing value.  In particular, booleans
    must be rejected despite being subclasses of Python's ``int``.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(TypeError, match="tested genus must be an integer"):
        knot.gilmer_genus_obstruction(genus)


def test_gilmer_genus_obstruction_rejects_negative_genus():
    """Reject negative integers after the separate integer-type validation.

    A surface genus is nonnegative, so accepting ``-1`` would make the rank
    eligibility condition and the reported ``genus+1`` lower bound
    mathematically meaningless.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(ValueError, match="must be nonnegative"):
        knot.gilmer_genus_obstruction(-1)


def test_genus_obstruction_results_are_immutable():
    """Freeze both the aggregate conclusion and each primary audit record.

    Genus-obstruction results are intended to serve as mathematical evidence.
    A caller must not be able to change a failed search into a certification,
    nor modify the status of an individual prime after the top-level result
    has been assembled.
    """
    result = GeneralizedAlgebraicKnot.torus_knot(2, 5).gilmer_genus_obstruction(0)

    # The conclusion stored on the aggregate result is read-only.
    with pytest.raises(FrozenInstanceError):
        result.certified = False

    # The nested per-prime explanation is independently read-only.
    with pytest.raises(FrozenInstanceError):
        result.primary_checks[0].certified = False


# ---------------------------------------------------------------------------
# Published genus-two computation
# ---------------------------------------------------------------------------

def test_published_genus_two_knot_satisfies_primary_gilmer_search():
    """Reproduce and audit the complete obstruction proving ``g_4^top >= 2``.

    This is the end-to-end regression for the implementation.  It verifies
    the theorem's exact eight-summand input, the two primary blocks, the size
    of each finite search, successful violation of Gilmer's inequality on
    every isotropic line, and internal consistency of a retained witness.
    Finally it raises the tested genus to demonstrate that the sufficient
    rank criterion becomes ineligible rather than overclaiming a stronger
    lower bound.
    """
    knot = _published_genus_two_knot()

    result = knot.gilmer_genus_obstruction(1)

    # These are the global conclusions needed for the paper's genus-two lower
    # bound: the classical signature correction vanishes, at least one primary
    # block certifies failure of genus one, and both 83 and 103 do so here.
    assert result.classical_signature == 0
    assert result.certified
    assert result.lower_bound == 2
    assert result.successful_primes == (83, 103)

    # Component order in the linking form must still match the knot's first
    # four q=83 summands followed by its four q=103 summands.  The character
    # witness tuples below rely on precisely this global ordering.
    assert result.linking_form.orders == (83,) * 4 + (103,) * 4

    for primary in result.primary_checks:
        prime = int(primary.prime)
        # The number of lines in projective 3-space over F_p is
        # p^3+p^2+p+1.  The split four-dimensional form has (p+1)^2
        # isotropic lines, and Lemma 3.1 requires a violation on every one.
        assert primary.projective_vectors_in_search_space == (
            prime ** 3 + prime ** 2 + prime + 1
        )

        # Unlike the projective-space count, this is the number of vectors
        # surviving the isotropy equation.  Every surviving line must have a
        # nonzero scalar multiple violating the inequality for certification.
        assert primary.isotropic_lines_examined == (prime + 1) ** 2
        assert primary.violating_lines == primary.isotropic_lines_examined
        assert primary.certified

        # A certified block has no counterexample line at which the search
        # stopped without finding a violation.
        assert primary.unresolved_isotropic_element is None

        # The search retains one representative violation for auditability.
        # Recompute its invariant through the public Casson--Gordon API so the
        # optimized signature lookup tables are checked against an independent
        # evaluation path.
        witness = primary.sample_witness
        invariant = knot.casson_gordon(witness.character_parameters)
        assert invariant.sigma == witness.sigma
        assert invariant.eta == witness.eta

        # Reconstruct both sides of Gilmer's inequality from the stored raw
        # quantities.  The strict comparison is the precise fact that makes
        # this character a violation rather than an equality case.
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
