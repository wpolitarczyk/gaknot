r"""Tests for Casson--Gordon invariants of ``(2,q)``-cables.

The numerical expectations below come directly from the two terms in
Marchwicka--Politarczyk, Lemma 2.12:

``-q + 2*a*(q-a)/q`` and ``2*sigma_K(a/q)``.

Companion signature values are chosen so they can be read independently from
the elementary jump sets of the small ``T(2,r)`` knots.  In particular, the
four-layer example checks the powers of two arising from successive cable
windings.  This is an important regression because the legacy implementation
used the linear layer numbers ``1,2,3,...`` instead.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import Integer, QQ

from gaknot import (
    CassonGordonInvariant,
    GeneralizedAlgebraicKnot,
    casson_gordon_invariant,
)


# ---------------------------------------------------------------------------
# The torus-pattern term
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "q, a, expected_pattern",
    [
        # The trivial character is a separate clause of Lemma 2.12; blindly
        # substituting a=0 into the nontrivial formula would give -q.
        (3, 0, QQ(0)),
        (3, 1, -QQ(5) / 3),
        (5, 1, -QQ(17) / 5),
        (5, 2, -QQ(13) / 5),
        (7, 1, -QQ(37) / 7),
        (7, 3, -QQ(25) / 7),
        (11, 5, -QQ(61) / 11),
    ],
)
def test_casson_gordon_torus_pattern_values(q, a, expected_pattern):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, q)

    result = casson_gordon_invariant(knot, a)
    summand = result.summands[0]

    assert isinstance(result, CassonGordonInvariant)
    assert summand.q == q
    assert summand.character_parameter == a
    assert summand.pattern_signature == expected_pattern
    assert summand.companion_signature == 0
    assert summand.satellite_signature == 0
    assert result.pattern_signature == expected_pattern
    assert result.satellite_signature == 0
    assert result.sigma == expected_pattern
    assert result.eta == 0
    # Exact output is part of the contract: genus comparisons must never pass
    # through a floating-point approximation.
    assert result.sigma in QQ


def test_casson_gordon_accepts_symmetric_torus_knot_spelling():
    # T(5,2) and T(2,5) are isotopic when there is no companion whose winding
    # convention needs to be retained.
    standard = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    symmetric = GeneralizedAlgebraicKnot.torus_knot(5, 2)

    assert symmetric.casson_gordon(2) == standard.casson_gordon(2)


@pytest.mark.parametrize("a", [-9, -4, 1, 6, 11, Integer(16)])
def test_casson_gordon_normalizes_character_parameter_modulo_q(a):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    result = knot.casson_gordon(a)

    assert result.character_parameters == (Integer(1),)
    assert result.sigma == -QQ(17) / 5


@pytest.mark.parametrize("q, a", [(3, 1), (5, 1), (5, 2), (7, 1), (7, 3)])
def test_casson_gordon_pattern_is_symmetric_under_character_conjugation(q, a):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, q)

    assert knot.casson_gordon(a).sigma == knot.casson_gordon(q - a).sigma


# ---------------------------------------------------------------------------
# Companion and iterated-cable contributions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "description, a, expected_companion, expected_pattern, expected_total",
    [
        # At theta=1/5, the trefoil signature is -2: precisely its jump at
        # 1/6 has been crossed.  The satellite contribution is therefore -4.
        (
            [(2, 3), (2, 5)],
            1,
            -2,
            -QQ(17) / 5,
            -QQ(37) / 5,
        ),
        # T(2,5) has crossed its first two positive-half jumps by theta=3/7,
        # so its Levine--Tristram signature is -4.
        (
            [(2, 5), (2, 7)],
            3,
            -4,
            -QQ(25) / 7,
            -QQ(81) / 7,
        ),
        # For the three-layer companion T(2,3;2,5;2,7), evaluation at 3/11
        # gives contributions -4, -4, and 0 at arguments 3/11, 6/11, and
        # 12/11 respectively.  The innermost argument is 2^2*theta, not the
        # legacy value 3*theta.
        (
            [(2, 3), (2, 5), (2, 7), (2, 11)],
            3,
            -8,
            -QQ(73) / 11,
            -QQ(249) / 11,
        ),
    ],
)
def test_casson_gordon_cable_values(
    description,
    a,
    expected_companion,
    expected_pattern,
    expected_total,
):
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(description)

    result = knot.casson_gordon(a)
    summand = result.summands[0]

    assert summand.companion_signature == expected_companion
    assert summand.satellite_signature == 2 * expected_companion
    assert summand.pattern_signature == expected_pattern
    assert summand.sigma == expected_total
    assert summand.eta == 0
    assert result.sigma == expected_total
    assert result.eta == 0


def test_casson_gordon_trivial_character_ignores_nontrivial_companion():
    # Even though the companion has a nonzero signature function, the
    # Casson--Gordon invariant and nullity vanish for chi_0.
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5), (2, 7)]
    )

    result = knot.casson_gordon(7)

    assert result.character_parameters == (0,)
    assert result.pattern_signature == 0
    assert result.satellite_signature == 0
    assert result.sigma == 0
    assert result.eta == 0


# ---------------------------------------------------------------------------
# Mirrors and connected sums
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "description, a",
    [
        ([(2, 5)], 2),
        ([(2, 3), (2, 5)], 1),
        ([(2, 5), (2, 7)], 3),
    ],
)
def test_casson_gordon_changes_sign_under_concordance_inverse(description, a):
    positive = GeneralizedAlgebraicKnot.iterated_torus_knot(description)
    negative = -positive

    positive_result = positive.casson_gordon(a)
    negative_result = negative.casson_gordon(a)

    assert negative_result.pattern_signature == -positive_result.pattern_signature
    assert negative_result.satellite_signature == -positive_result.satellite_signature
    assert negative_result.sigma == -positive_result.sigma
    # Nullity is a dimension and therefore does not change under mirroring.
    assert negative_result.eta == positive_result.eta


def test_casson_gordon_connected_sum_signature_and_nullity():
    knot = (
        GeneralizedAlgebraicKnot.torus_knot(2, 5)
        - GeneralizedAlgebraicKnot.torus_knot(2, 5)
    )

    result = knot.casson_gordon([1, 1])

    # Signature contributions cancel, while Proposition 2.8 contributes one
    # to eta because both restrictions of the character are nontrivial.
    assert len(result.summands) == 2
    assert result.character_parameters == (1, 1)
    assert result.pattern_signature == 0
    assert result.satellite_signature == 0
    assert result.sigma == 0
    assert result.eta == 1


@pytest.mark.parametrize(
    "parameters, expected_eta",
    [
        ([0, 0, 0], 0),
        ([1, 0, 0], 0),
        ([1, 1, 0], 1),
        ([1, 1, 1], 2),
    ],
)
def test_casson_gordon_connected_sum_nullity_counts_nontrivial_restrictions(
    parameters,
    expected_eta,
):
    trefoil = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    knot = trefoil + trefoil + trefoil

    assert knot.casson_gordon(parameters).eta == expected_eta


def test_casson_gordon_result_is_immutable():
    result = GeneralizedAlgebraicKnot.torus_knot(2, 5).casson_gordon(1)

    with pytest.raises(FrozenInstanceError):
        result.summands = ()
    with pytest.raises(FrozenInstanceError):
        result.summands[0].pattern_signature = 0


# ---------------------------------------------------------------------------
# Public validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_knot", [None, "T(2,5)", [(2, 5)]])
def test_casson_gordon_rejects_non_knot_objects(bad_knot):
    with pytest.raises(TypeError, match="Expected a GeneralizedAlgebraicKnot"):
        casson_gordon_invariant(bad_knot, 1)


@pytest.mark.parametrize("bad_parameter", [True, False, QQ(1) / 5, 1.0, "1", None])
def test_casson_gordon_rejects_noninteger_character_parameter(bad_parameter):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(TypeError, match="Character parameter at component 0"):
        knot.casson_gordon([bad_parameter])


@pytest.mark.parametrize(
    "parameters, expected_count, received_count",
    [
        ([], 1, 0),
        ([1, 2], 1, 2),
    ],
)
def test_casson_gordon_rejects_wrong_parameter_count(
    parameters,
    expected_count,
    received_count,
):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(
        ValueError,
        match=f"Expected {expected_count}.*received {received_count}",
    ):
        knot.casson_gordon(parameters)


def test_casson_gordon_requires_sequence_for_connected_sum():
    cinquefoil = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    knot = cinquefoil + cinquefoil

    with pytest.raises(TypeError, match="one integer per summand"):
        knot.casson_gordon(1)


def test_casson_gordon_rejects_nonprime_outer_order():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 9)

    with pytest.raises(ValueError, match="q=9.*requires q prime"):
        knot.casson_gordon(1)


def test_casson_gordon_rejects_outer_winding_other_than_two():
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (3, 5)]
    )

    with pytest.raises(ValueError, match="outer winding number 2"):
        knot.casson_gordon(1)


def test_casson_gordon_does_not_swap_parameters_of_a_genuine_cable():
    # Although T(5,2) is isotopic to T(2,5) as an ordinary pattern knot, these
    # pairs encode winding numbers five and two when used as cable operators.
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (5, 2)]
    )

    with pytest.raises(ValueError, match="outer winding number 2"):
        knot.casson_gordon(1)
