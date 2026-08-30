r"""Tests for Casson--Gordon invariants of ``(2,q)``-cables.

The numerical expectations below come directly from the two terms in
Marchwicka--Politarczyk, Lemma 2.12:

``-q + 2*a*(q-a)/q`` and ``2*sigma_K(a/q)``.

Companion signature values are chosen so they can be read independently from
the elementary jump sets of the small ``T(2,r)`` knots.  In particular, the
four-layer example checks the powers of two arising from successive cable
windings.  This is an important regression because the legacy implementation
used the linear layer numbers ``1,2,3,...`` instead.

The suite is organized around the layers of the public contract:

* the first group isolates the torus-pattern term and checks exact arithmetic,
  normalization of character parameters, and conjugation symmetry;
* the second group adds nontrivial companions and checks that the
  Levine--Tristram value is evaluated at the correct cabling arguments;
* the third group checks how signed summands are assembled into concordance
  inverses and connected sums, including the connected-sum correction to the
  nullity; and
* the final group exercises every boundary of the currently supported domain.

Most tests inspect both the total ``CassonGordonInvariant`` and its individual
``CassonGordonSummand`` record.  This is intentional: equality of the final
number alone would not detect a bug that accidentally moved a contribution
between the pattern and companion terms, even though downstream diagnostic
code relies on that decomposition.
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
        # For every remaining row, evaluate -q + 2*a*(q-a)/q exactly.  The
        # small q=3 case gives -3 + 4/3 = -5/3.
        (3, 1, -QQ(5) / 3),
        # Two characters for q=5 distinguish adjacent values of ``a`` rather
        # than testing only a single point of the quadratic pattern term.
        (5, 1, -QQ(17) / 5),
        (5, 2, -QQ(13) / 5),
        # The q=7 rows play the same role at a larger prime and include the
        # character immediately below the conjugation midpoint.
        (7, 1, -QQ(37) / 7),
        (7, 3, -QQ(25) / 7),
        # A final q=11 value guards against code that happens to work only for
        # the very smallest primes used elsewhere in the unit tests.
        (11, 5, -QQ(61) / 11),
    ],
)
def test_casson_gordon_torus_pattern_values(q, a, expected_pattern):
    """Check Lemma 2.12 directly when the companion is the unknot.

    Each row supplies an outer prime ``q`` and a distinguished character
    parameter ``a``.  Since ``T(2,q)`` is the ``(2,q)`` pattern applied to the
    unknot, the companion contribution must be zero and the total invariant
    must equal the exact rational pattern term.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, q)

    result = casson_gordon_invariant(knot, a)
    summand = result.summands[0]

    # The free function is part of the public API, so first verify that it
    # returns the advertised structured result rather than a bare number.
    assert isinstance(result, CassonGordonInvariant)

    # The summand record must retain the validated outer order and the
    # normalized character parameter.  These fields are later reused by the
    # genus-obstruction code when it organizes components by primary order.
    assert summand.q == q
    assert summand.character_parameter == a

    # With the unknot as companion, every satellite/companion term vanishes.
    # Consequently the pattern contribution is visible unchanged both on the
    # summand and after aggregation by the top-level result.
    assert summand.pattern_signature == expected_pattern
    assert summand.companion_signature == 0
    assert summand.satellite_signature == 0
    assert result.pattern_signature == expected_pattern
    assert result.satellite_signature == 0
    assert result.sigma == expected_pattern

    # A single prime-order nontrivial restriction has no connected-sum
    # correction, and the individual cable nullity is zero in this domain.
    assert result.eta == 0

    # Exact output is part of the contract: genus comparisons must never pass
    # through a floating-point approximation.
    assert result.sigma in QQ


def test_casson_gordon_accepts_symmetric_torus_knot_spelling():
    """Accept ``T(q,2)`` only in the one-layer torus-knot situation.

    This is the positive half of the winding-convention tests.  A plain torus
    knot is unchanged when its two parameters are exchanged, so both valid
    descriptions must produce the same structured invariant.
    """
    # T(5,2) and T(2,5) are isotopic when there is no companion whose winding
    # convention needs to be retained.
    standard = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    symmetric = GeneralizedAlgebraicKnot.torus_knot(5, 2)

    assert symmetric.casson_gordon(2) == standard.casson_gordon(2)


@pytest.mark.parametrize("a", [-9, -4, 1, 6, 11, Integer(16)])
def test_casson_gordon_normalizes_character_parameter_modulo_q(a):
    """Reduce Python and Sage integer parameters to their class in ``Z/q``.

    Every listed input represents ``1 mod 5``.  The test checks both the
    stored canonical representative and the computed invariant, thereby
    ruling out an implementation that normalizes only for display after
    evaluating the formula with the original integer.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    result = knot.casson_gordon(a)

    # ``character_parameters`` exposes a tuple even for one summand so that
    # consumers can handle one- and many-component results uniformly.
    assert result.character_parameters == (Integer(1),)
    assert result.sigma == -QQ(17) / 5


@pytest.mark.parametrize("q, a", [(3, 1), (5, 1), (5, 2), (7, 1), (7, 3)])
def test_casson_gordon_pattern_is_symmetric_under_character_conjugation(q, a):
    """Verify invariance under replacing ``chi_a`` by ``chi_{q-a}``.

    Complex conjugation changes the character parameter from ``a`` to
    ``-a mod q``.  The pattern formula contains ``a(q-a)`` and must therefore
    give the same signature to both restrictions.
    """
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
    """Separate and verify the pattern and companion terms of cable examples.

    The parametrized examples increase the depth of the companion.  Besides
    checking the final Casson--Gordon signature, the assertions ensure that
    the raw Levine--Tristram value is recorded separately and is multiplied
    by exactly two when it enters the satellite formula.
    """
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(description)

    result = knot.casson_gordon(a)
    summand = result.summands[0]

    # ``companion_signature`` is the signed raw Levine--Tristram value.
    # ``satellite_signature`` is the doubled term that actually occurs in
    # Lemma 2.12.  Testing both catches either a missing factor of two or an
    # accidental factor applied twice.
    assert summand.companion_signature == expected_companion
    assert summand.satellite_signature == 2 * expected_companion

    # The expected pattern value is computed independently from the torus
    # term.  The summand total must then be the sum of these two pieces.
    assert summand.pattern_signature == expected_pattern
    assert summand.sigma == expected_total

    # Prime-order roots cannot be zeros of a knot Alexander polynomial, so
    # the supported individual cable nullity is zero.  A one-summand result
    # introduces no additional connected-sum nullity either.
    assert summand.eta == 0
    assert result.sigma == expected_total
    assert result.eta == 0


def test_casson_gordon_trivial_character_ignores_nontrivial_companion():
    """Use the special trivial-character clause instead of the displayed formula.

    The input ``7`` is zero in the outer ``Z/7`` character group.  The knot
    deliberately has a nontrivial iterated companion, so a nonzero answer
    would reveal that the code evaluated companion signatures before handling
    the trivial restriction.
    """
    # Even though the companion has a nonzero signature function, the
    # Casson--Gordon invariant and nullity vanish for chi_0.
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5), (2, 7)]
    )

    result = knot.casson_gordon(7)

    # Check every exposed contribution, not just the total: all data attached
    # to the trivial restriction must be zero by the separate clause of the
    # cabling formula.
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
    """Check sign reversal for signatures and sign invariance for nullities.

    Negating a GA-knot records its concordance inverse by reversing the sign
    of the visible summand.  Signature-type quantities are signed, whereas a
    nullity is a dimension and therefore cannot become negative.
    """
    positive = GeneralizedAlgebraicKnot.iterated_torus_knot(description)
    negative = -positive

    positive_result = positive.casson_gordon(a)
    negative_result = negative.casson_gordon(a)

    # Test the two contributions separately before testing their sum.  This
    # prevents two independent sign errors from cancelling in ``sigma``.
    assert negative_result.pattern_signature == -positive_result.pattern_signature
    assert negative_result.satellite_signature == -positive_result.satellite_signature
    assert negative_result.sigma == -positive_result.sigma
    # Nullity is a dimension and therefore does not change under mirroring.
    assert negative_result.eta == positive_result.eta


def test_casson_gordon_connected_sum_signature_and_nullity():
    """Exercise additivity and the nullity correction on ``K # -K``.

    The two signature contributions cancel because the structural model keeps
    the positive and negative summands visible.  Both character restrictions
    are nevertheless nontrivial, so the connected-sum formula contributes
    ``r-1 = 1`` to the nullity.
    """
    knot = (
        GeneralizedAlgebraicKnot.torus_knot(2, 5)
        - GeneralizedAlgebraicKnot.torus_knot(2, 5)
    )

    result = knot.casson_gordon([1, 1])

    # Signature contributions cancel, while Proposition 2.8 contributes one
    # to eta because both restrictions of the character are nontrivial.
    # Retaining two summands is itself part of the structural API: the package
    # must not simplify K # -K before evaluating restricted characters.
    assert len(result.summands) == 2
    assert result.character_parameters == (1, 1)

    # Pattern terms cancel pairwise and there are no nontrivial companions in
    # this example, so the two aggregate contribution fields and their sum
    # all vanish independently.
    assert result.pattern_signature == 0
    assert result.satellite_signature == 0
    assert result.sigma == 0
    assert result.eta == 1


@pytest.mark.parametrize(
    "parameters, expected_eta",
    [
        # With zero or one nontrivial restriction, max(r-1,0) is zero.
        ([0, 0, 0], 0),
        ([1, 0, 0], 0),
        # Two and three nontrivial restrictions contribute one and two.
        ([1, 1, 0], 1),
        ([1, 1, 1], 2),
    ],
)
def test_casson_gordon_connected_sum_nullity_counts_nontrivial_restrictions(
    parameters,
    expected_eta,
):
    """Check the ``max(r-1, 0)`` connected-sum nullity correction.

    Each trefoil summand has zero individual nullity.  Varying only the number
    ``r`` of nontrivial restrictions therefore isolates the correction term:
    no correction for ``r=0`` or ``r=1``, then one and two for ``r=2`` and
    ``r=3`` respectively.
    """
    trefoil = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    knot = trefoil + trefoil + trefoil

    assert knot.casson_gordon(parameters).eta == expected_eta


def test_casson_gordon_result_is_immutable():
    """Prevent callers from corrupting either level of returned provenance.

    Both dataclasses are frozen.  The first assignment targets the aggregate
    result; the second reaches into its tuple and targets the per-summand
    record.  Testing both levels ensures that cached mathematical evidence
    cannot be changed after construction.
    """
    result = GeneralizedAlgebraicKnot.torus_knot(2, 5).casson_gordon(1)

    # Replacing the collection of contributions must fail.
    with pytest.raises(FrozenInstanceError):
        result.summands = ()

    # Mutating a field inside an existing contribution must fail as well.
    with pytest.raises(FrozenInstanceError):
        result.summands[0].pattern_signature = 0


# ---------------------------------------------------------------------------
# Public validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_knot", [None, "T(2,5)", [(2, 5)]])
def test_casson_gordon_rejects_non_knot_objects(bad_knot):
    """Require the structural GA-knot model at the free-function boundary.

    ``None``, a human-readable name, and a raw cable description are plausible
    accidental inputs.  Rejecting all three avoids silently inventing signs,
    component boundaries, or cabling conventions.
    """
    with pytest.raises(TypeError, match="Expected a GeneralizedAlgebraicKnot"):
        casson_gordon_invariant(bad_knot, 1)


@pytest.mark.parametrize("bad_parameter", [True, False, QQ(1) / 5, 1.0, "1", None])
def test_casson_gordon_rejects_noninteger_character_parameter(bad_parameter):
    """Accept exact integers only as distinguished character parameters.

    The cases cover Python booleans (which otherwise subclass ``int``), an
    exact nonintegral Sage rational, a floating-point integral value, a string,
    and a missing value.  The component index in the message is checked
    because it is essential when diagnosing a long connected sum.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(TypeError, match="Character parameter at component 0"):
        knot.casson_gordon([bad_parameter])


@pytest.mark.parametrize(
    "parameters, expected_count, received_count",
    [
        # Missing the only required component parameter.
        ([], 1, 0),
        # Supplying an additional parameter for a nonexistent component.
        ([1, 2], 1, 2),
    ],
)
def test_casson_gordon_rejects_wrong_parameter_count(
    parameters,
    expected_count,
    received_count,
):
    """Require exactly one character restriction per visible summand.

    Both an omitted restriction and an extra restriction are tested.  The
    exception must report expected and received counts so the ordering error
    is actionable for users constructing connected sums programmatically.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)

    with pytest.raises(
        ValueError,
        match=f"Expected {expected_count}.*received {received_count}",
    ):
        knot.casson_gordon(parameters)


def test_casson_gordon_requires_sequence_for_connected_sum():
    """Reject an ambiguous scalar parameter for a multi-summand knot.

    Scalar shorthand is intentionally limited to the one-summand case.  For a
    connected sum the caller must spell out one restriction per structural
    summand, making the component-to-character correspondence explicit.
    """
    cinquefoil = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    knot = cinquefoil + cinquefoil

    with pytest.raises(TypeError, match="one integer per summand"):
        knot.casson_gordon(1)


def test_casson_gordon_rejects_nonprime_outer_order():
    """Enforce the prime-``q`` hypothesis of the implemented formula.

    ``T(2,9)`` is a perfectly valid knot, so construction succeeds; only the
    Casson--Gordon method must reject it because the current implementation
    and its distinguished primary-character model assume prime outer order.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 9)

    with pytest.raises(ValueError, match="q=9.*requires q prime"):
        knot.casson_gordon(1)


def test_casson_gordon_rejects_outer_winding_other_than_two():
    """Reject genuine cables outside the implemented winding-two family.

    The inner trefoil is supported, but the outer pair ``(3,5)`` describes a
    winding-three satellite.  This isolates validation of the outer cabling
    operation from validation of the companion itself.
    """
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (3, 5)]
    )

    with pytest.raises(ValueError, match="outer winding number 2"):
        knot.casson_gordon(1)


def test_casson_gordon_does_not_swap_parameters_of_a_genuine_cable():
    """Do not apply torus-knot symmetry to an outer satellite operator.

    This is the negative counterpart to the one-layer ``T(5,2)`` test above.
    Once a companion is present, the first entry of ``(p,q)`` is the winding
    number.  Swapping ``(5,2)`` to ``(2,5)`` would silently change the cable,
    so the unsupported winding-five description must be rejected.
    """
    # Although T(5,2) is isotopic to T(2,5) as an ordinary pattern knot, these
    # pairs encode winding numbers five and two when used as cable operators.
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (5, 2)]
    )

    with pytest.raises(ValueError, match="outer winding number 2"):
        knot.casson_gordon(1)
