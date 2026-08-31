"""Regression tests for the fixed twisted Alexander representative.

The invariant is defined only up to multiplication by Laurent-ring units, but
``twisted_alexander_torus_knot`` deliberately returns the exact representative
from Proposition 3.3.  Exact equalities below test that public normalization
contract in addition to the underlying invariant class.
"""

import pytest
from sage.all import QQ, CyclotomicField, PolynomialRing

from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.character import Character
from gaknot.invariants.torus_character import (
    TorusCharacterOrbit,
    torus_character_orbit,
    torus_pattern_phase_orbit,
)
from gaknot.invariants.twisted_alexander import twisted_alexander_torus_knot


def _parse_expected_representative(q, expression):
    """Parse a displayed formula in the same Q(zeta_q)(t) field as the result.

    Sage names the canonical generator of ``CyclotomicField(q)`` ``zeta<q>``.
    The fraction-field parser can therefore resolve both ``t`` and coefficients
    such as ``zeta5^3`` from the ring itself; no Python ``eval`` or dynamically
    created local variables are involved.
    """
    coefficient_field = CyclotomicField(q)
    polynomial_ring = PolynomialRing(coefficient_field, 't')
    return polynomial_ring.fraction_field()(expression)


# --- Shared character-orbit conversion --------------------------------------

# The twisted Alexander denominator, Yanagida's matrices, and the satellite
# phase shifts must all use the same cyclic ordering of the character orbit.
# These tests therefore exercise the extracted basis-conversion helper
# directly, in addition to the polynomial regression tests below.  A cyclic
# rotation often describes an isomorphic representation, but the public API
# promises a deterministic representative, so exact tuple equality matters.
@pytest.mark.parametrize(
    "p, q, generator_values, expected_orbit",
    [
        (2, 3, [QQ(1) / 3], (2, 1)),
        (2, 5, [QQ(1) / 5], (4, 1)),
        (3, 4, [QQ(1) / 4, 0], (3, 1, 0)),
        (3, 5, [QQ(1) / 5, QQ(2) / 5], (1, 2, 2)),
    ],
)
def test_torus_character_orbit_uses_the_historical_smith_basis_order(
    p, q, generator_values, expected_orbit
):
    """Recover exact deck-orbit coordinates from Smith-generator values.

    The inputs are character images in the public homology basis.  The
    expected outputs were already implicit in the exact twisted Alexander
    representatives tested later in this file.  Isolating them here makes the
    basis change itself observable and allows the signature code to reuse it
    without reverse-engineering a polynomial denominator.
    """
    orbit = torus_character_orbit(p, q, generator_values)

    assert isinstance(orbit, TorusCharacterOrbit)
    assert orbit.p == p
    assert orbit.q == q
    assert orbit.generator_values == tuple(generator_values)
    assert orbit.a_values == expected_orbit

    # The determinant/deck-orbit relation is exact in Z/qZ.  Phase arguments
    # are exact rational numbers, never floating approximations to roots of
    # unity, and preserve the same order as the integer orbit.
    assert sum(orbit.a_values) % q == 0
    assert orbit.phase_arguments == tuple(QQ(a) / q for a in expected_orbit)

    # The general cable-phase implementation uses arbitrary cover degree and
    # arbitrary Smith factors.  On this older, specialized n=p domain it must
    # nevertheless reproduce the established orbit exactly.  This comparison
    # prevents the two consumers of the distinguished class--twisted
    # Alexander polynomials and Theorem 4.19 phases--from drifting to different
    # Smith bases or cyclic starting points.
    phase_orbit = torus_pattern_phase_orbit(
        p,
        q,
        p,
        generator_values,
    )
    assert phase_orbit.smith_factors == tuple(q for _ in range(p - 1))
    assert phase_orbit.phase_arguments == orbit.phase_arguments


def test_torus_character_orbit_normalizes_character_and_orbit_representatives():
    """Normalize two equivalent descriptions without changing their classes."""
    # 6/5 and 1/5 define the same element of Q/Z.  The computed orbit is also
    # returned in the canonical integer range 0,...,q-1.
    orbit = torus_character_orbit(2, 5, [QQ(6) / 5])

    assert orbit.generator_values == (QQ(1) / 5,)
    assert orbit.a_values == (4, 1)
    assert all(0 <= value < 5 for value in orbit.a_values)


def test_torus_character_orbit_of_zero_character_is_zero():
    """Check that every deck translate evaluates trivially under chi=0."""
    orbit = torus_character_orbit(5, 2, [0, 0, 0, 0])

    assert orbit.a_values == (0, 0, 0, 0, 0)
    assert orbit.phase_arguments == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    "p, q, generator_values, error_type, message",
    [
        (True, 5, [0], TypeError, "p must be an integer"),
        (1, 5, [], ValueError, "p must be greater than one"),
        (2, 4, [0], ValueError, "p and q must be relatively prime"),
        (3, 4, [0], ValueError, "exactly p-1=2 entries"),
        (2, 5, [0.2], TypeError, "exact rational number"),
        (2, 5, [QQ(1) / 3], ValueError, "q-torsion"),
    ],
)
def test_torus_character_orbit_rejects_incompatible_public_data(
    p, q, generator_values, error_type, message
):
    """Reject malformed coordinates before they enter a matrix computation."""
    with pytest.raises(error_type, match=message):
        torus_character_orbit(p, q, generator_values)


# --- Exact formula for the trivial character ---------------------------------

# With every a_j equal to zero, the denominator in Proposition 3.3 is
# (t - 1)^p.  These cases test the resulting cancellations for several values
# and both parameter orderings, since p determines the cover while q determines
# the coefficient field.
@pytest.mark.parametrize(
    "p, q, expected_representative_str",
    [
        (2, 3, "(-t^2 - t - 1)/(t - 1)"),
        (2, 5, "(-t^4 - t^3 - t^2 - t - 1)/(t - 1)"),
        (3, 2, "(t^2 + 2*t + 1)/(t - 1)"),
        (
            3,
            4,
            "(t^6 + 2*t^5 + 3*t^4 + 4*t^3 + 3*t^2 + 2*t + 1)"
            "/(t - 1)",
        ),
        (5, 2, "(t^4 + 4*t^3 + 6*t^2 + 4*t + 1)/(t - 1)"),
        (2, 7, "(-t^6 - t^5 - t^4 - t^3 - t^2 - t - 1)/(t - 1)"),
        (
            4,
            3,
            "(-t^6 - 3*t^5 - 6*t^4 - 7*t^3 - 6*t^2 - 3*t - 1)"
            "/(t - 1)",
        ),
        (
            5,
            4,
            "(t^12 + 4*t^11 + 10*t^10 + 20*t^9 + 31*t^8 + 40*t^7 "
            "+ 44*t^6 + 40*t^5 + 31*t^4 + 20*t^3 + 10*t^2 + 4*t + 1)"
            "/(t - 1)",
        ),
        (
            3,
            5,
            "(t^8 + 2*t^7 + 3*t^6 + 4*t^5 + 5*t^4 + 4*t^3 "
            "+ 3*t^2 + 2*t + 1)/(t - 1)",
        ),
        (2, 9, "(-t^8 - t^7 - t^6 - t^5 - t^4 - t^3 - t^2 - t - 1)/(t - 1)"),
    ],
)
def test_twisted_alexander_torus_knot_trivial(
    p, q, expected_representative_str
):
    """Check the fixed Proposition 3.3 representative for the zero character."""
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q)
    h1 = BranchedCoverHomology(knot, p)
    factor_count = len(h1.decomposition[0]['layers'][0]['base_factors'])
    character = Character(h1, [[[0] * factor_count]])

    result = twisted_alexander_torus_knot(knot, character)
    expected = _parse_expected_representative(q, expected_representative_str)

    assert result == expected


# --- Exact formula for nontrivial characters ---------------------------------

# Here x_values are the images of the Smith-basis generators of
# H_1(Sigma_p(T(p,q))) = (Z/qZ)^(p-1).  The implementation converts them to the
# orbit values a_j appearing as exponents of zeta_q in Proposition 3.3.
@pytest.mark.parametrize(
    "p, q, x_values, expected_representative_str",
    [
        (2, 3, [QQ(1) / 3], "-t + 1"),
        (
            2,
            5,
            [QQ(1) / 5],
            "-t^3 + (zeta5^3 + zeta5^2 + 1)*t^2 "
            "+ (-zeta5^3 - zeta5^2 - 1)*t + 1",
        ),
        (3, 2, [QQ(1) / 2, QQ(1) / 2], "t - 1"),
        (2, 3, [QQ(2) / 3], "-t + 1"),
        (3, 4, [QQ(1) / 4, 0], "t^5 + t^4 - t - 1"),
        (
            2,
            7,
            [QQ(1) / 7],
            "-t^5 + (zeta7^5 + zeta7^4 + zeta7^3 + zeta7^2 + 1)*t^4 "
            "+ (-zeta7^5 - zeta7^2 - 1)*t^3 "
            "+ (zeta7^5 + zeta7^2 + 1)*t^2 "
            "+ (-zeta7^5 - zeta7^4 - zeta7^3 - zeta7^2 - 1)*t + 1",
        ),
        (
            4,
            3,
            [QQ(1) / 3, 0, 0],
            "-t^5 - t^4 - t^3 + t^2 + t + 1",
        ),
        (
            5,
            2,
            [QQ(1) / 2, QQ(1) / 2, QQ(1) / 2, QQ(1) / 2],
            "t^3 - 3*t^2 + 3*t - 1",
        ),
        (
            2,
            9,
            [QQ(1) / 9],
            "-t^7 + (zeta9^5 + zeta9^2 - zeta9)*t^6 "
            "+ (zeta9^4 - zeta9^2 + zeta9 - 1)*t^5 "
            "+ (zeta9^5 + zeta9^2 - zeta9 + 1)*t^4 "
            "+ (-zeta9^5 - zeta9^2 + zeta9 - 1)*t^3 "
            "+ (-zeta9^4 + zeta9^2 - zeta9 + 1)*t^2 "
            "+ (-zeta9^5 - zeta9^2 + zeta9)*t + 1",
        ),
        (
            3,
            5,
            [QQ(1) / 5, QQ(2) / 5],
            "t^7 + (zeta5^3 - zeta5^2 - zeta5 - 1)*t^6 "
            "+ (zeta5^3 + 2*zeta5^2 + 3*zeta5)*t^5 "
            "+ (-4*zeta5^3 - 3*zeta5^2 - 2*zeta5 - 1)*t^4 "
            "+ (zeta5^3 + 2*zeta5^2 - 2*zeta5 - 1)*t^3 "
            "+ (zeta5^3 + 2*zeta5^2 + 3*zeta5 + 3)*t^2 "
            "+ (-2*zeta5^2 - zeta5)*t - 1",
        ),
    ],
)
def test_twisted_alexander_torus_knot_nontrivial(
    p, q, x_values, expected_representative_str
):
    """Check exact representatives for characters with nonzero generator images."""
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q)
    h1 = BranchedCoverHomology(knot, p)
    character = Character(h1, [[x_values]])

    result = twisted_alexander_torus_knot(knot, character)
    expected = _parse_expected_representative(q, expected_representative_str)

    assert result == expected


# --- Character/knot domain association ---------------------------------------

@pytest.mark.parametrize("character_value", [QQ(0), QQ(1) / 5])
def test_twisted_alexander_rejects_character_from_different_knot(character_value):
    # Both knots use their double branched covers and both homology groups have
    # one generator.  Matching those superficial properties is insufficient:
    # the character must use the Smith basis belonging to the supplied knot.
    target_knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    character_knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    character_h1 = BranchedCoverHomology(character_knot, 2)
    character = Character(character_h1, [[[character_value]]])

    # The zero character used to be silently accepted, while the nonzero
    # character failed later during an unrelated rational-to-integer coercion.
    # Both must now fail at the domain boundary with the same clear error.
    with pytest.raises(
        ValueError,
        match="Character must be defined on the homology of the supplied knot",
    ):
        twisted_alexander_torus_knot(target_knot, character)


def test_twisted_alexander_accepts_equivalent_knot_instance():
    # Knot association is structural, not an object-identity requirement.  A
    # character built from an independently constructed T(2,3) is valid for a
    # second T(2,3) instance with the same description.
    target_knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    equivalent_knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    character = Character(BranchedCoverHomology(equivalent_knot, 2), [[[0]]])

    expected = _parse_expected_representative(
        3,
        "(-t^2 - t - 1)/(t - 1)",
    )

    assert twisted_alexander_torus_knot(target_knot, character) == expected


# --- Direct-function input validation ----------------------------------------

@pytest.fixture
def positive_trefoil():
    """Return the positive torus knot T(2,3) used by validation tests."""
    return GeneralizedAlgebraicKnot.torus_knot(2, 3)


def test_twisted_alexander_rejects_iterated_torus_knot():
    # Proposition 3.3 applies to a single torus knot, not to an iterated cable.
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5)])
    character = Character(BranchedCoverHomology(knot, 2), [[[0], []]])

    with pytest.raises(ValueError, match="Knot must be a positive torus knot"):
        twisted_alexander_torus_knot(knot, character)


def test_twisted_alexander_rejects_negative_torus_knot():
    # Use a character from the negative knot itself so the test isolates the
    # unsupported orientation rather than also supplying a mismatched domain.
    negative_trefoil = GeneralizedAlgebraicKnot.torus_knot(2, 3, sign=-1)
    character = Character(
        BranchedCoverHomology(negative_trefoil, 2),
        [[[0]]],
    )

    with pytest.raises(
        NotImplementedError,
        match="negative torus knots not yet implemented",
    ):
        twisted_alexander_torus_knot(negative_trefoil, character)


@pytest.mark.parametrize("cover_degree", [3, 10], ids=["degree-three", "degree-ten"])
def test_twisted_alexander_rejects_wrong_cover_degree(
    positive_trefoil, cover_degree
):
    # For T(2,3), the formula requires a character on the p=2 cover regardless
    # of whether another cover happens to have a compatible-looking homology.
    h1 = BranchedCoverHomology(positive_trefoil, cover_degree)
    zero_values = [0] * len(h1.all_invariant_factors)
    character = Character(h1, [[zero_values]])

    with pytest.raises(
        ValueError,
        match="Formula requires character on the 2-fold cover",
    ):
        twisted_alexander_torus_knot(positive_trefoil, character)


@pytest.mark.parametrize(
    "invalid_character",
    [None, "not a character"],
    ids=["none", "string"],
)
def test_twisted_alexander_rejects_non_character(
    positive_trefoil, invalid_character
):
    # Fail at the public type boundary instead of producing an attribute error
    # while trying to read homology data from an arbitrary object.
    with pytest.raises(TypeError, match="Expected a Character object"):
        twisted_alexander_torus_knot(positive_trefoil, invalid_character)


def test_twisted_alexander_rejects_connected_sum():
    # A connected sum has multiple torus-knot components and is outside the
    # single-summand hypothesis of the implemented formula.
    trefoil = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    cinquefoil = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    knot_sum = trefoil + cinquefoil
    character = Character(
        BranchedCoverHomology(knot_sum, 2),
        [[[0]], [[0]]],
    )

    with pytest.raises(ValueError, match="Knot must be a positive torus knot"):
        twisted_alexander_torus_knot(knot_sum, character)


# --- Character convenience method --------------------------------------------

# The method should add no new normalization or computation; it obtains the
# knot from ``character.homology`` and delegates to the direct function.
@pytest.mark.parametrize(
    "p, q, x_val",
    [
        (2, 3, QQ(1) / 3),
        (2, 5, QQ(1) / 5),
        (3, 2, QQ(1) / 2),
        (2, 7, QQ(1) / 7),
        (5, 2, QQ(1) / 2),
        (4, 3, QQ(1) / 3),
        (5, 4, QQ(1) / 4),
        (3, 5, QQ(2) / 5),
        (2, 9, QQ(1) / 9),
        (2, 11, QQ(1) / 11),
    ],
)
def test_character_method_twisted_alexander(p, q, x_val):
    """Require the Character method to return the direct function's result."""
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q)
    h1 = BranchedCoverHomology(knot, p)
    factor_count = len(h1.decomposition[0]['layers'][0]['base_factors'])
    character = Character(h1, [[[x_val] * factor_count]])

    # The convenience method must delegate using the knot and character stored
    # in this homology object, without changing the chosen representative.
    method_result = character.twisted_alexander_polynomial()
    direct_result = twisted_alexander_torus_knot(knot, character)

    assert method_result == direct_result


def test_character_method_rejects_iterated_torus_knot():
    # The convenience method obtains its knot from the character's homology,
    # but it retains the direct function's single-torus-knot restriction.
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5)])
    character = Character(BranchedCoverHomology(knot, 2), [[[0], []]])

    with pytest.raises(
        NotImplementedError,
        match="currently only implemented for positive torus knots",
    ):
        character.twisted_alexander_polynomial()


def test_character_method_rejects_connected_sum():
    # Connected sums are also outside the method's current implementation
    # domain, even when every summand is itself a positive torus knot.
    trefoil = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    cinquefoil = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    knot_sum = trefoil + cinquefoil
    character = Character(
        BranchedCoverHomology(knot_sum, 2),
        [[[0]], [[0]]],
    )

    with pytest.raises(
        NotImplementedError,
        match="currently only implemented for positive torus knots",
    ):
        character.twisted_alexander_polynomial()


def test_character_method_rejects_negative_torus_knot():
    # Mirroring changes the sign stored in the knot description; the current
    # convenience method deliberately declines that case.
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3, sign=-1)
    character = Character(BranchedCoverHomology(knot, 2), [[[0]]])

    with pytest.raises(
        NotImplementedError,
        match="currently only implemented for positive torus knots",
    ):
        character.twisted_alexander_polynomial()


def test_character_method_supports_large_p_small_q():
    # T(7,2) exercises a seven-dimensional representation while keeping the
    # coefficient field rational because the second parameter is two.
    knot = GeneralizedAlgebraicKnot.torus_knot(7, 2)
    character = Character(BranchedCoverHomology(knot, 7), [[[0] * 6]])

    method_result = character.twisted_alexander_polynomial()
    direct_result = twisted_alexander_torus_knot(knot, character)

    assert method_result == direct_result


def test_character_method_rejects_wrong_cover_degree():
    # The method delegates to the same Proposition 3.3 implementation, so the
    # character must still live on the p-fold cover of T(p,q).
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    character = Character(BranchedCoverHomology(knot, 4), [[[0]]])

    with pytest.raises(
        ValueError,
        match="Formula requires character on the 2-fold cover",
    ):
        character.twisted_alexander_polynomial()
