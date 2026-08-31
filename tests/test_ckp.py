#!/usr/bin/env sage -python

r"""Tests for the Conway--Kim--Politarczyk reconstruction layer.

The paper's Section 3 is a chain of exact implications rather than a black-box
polynomial formula.  A character orbit determines two representation matrices;
the matrices satisfy the torus-knot group relation; two Fox determinants are
computed from them; and their quotient gives the exterior and zero-surgery
twisted Alexander representatives.  These tests retain the same separation.
If a convention changes, the failing test should identify the first affected
mathematical step instead of merely reporting a different final expression.

The final group of tests covers the ``s``-levels in Proposition 5.4.  The
worked four-summand example has cancellations at both levels, so it checks the
source-layer order, formal signs, substitution powers, and root moduli used by
the proof's root-separation argument.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import QQ, identity_matrix, matrix

from gaknot import (
    BranchedCoverHomology,
    Character,
    GeneralizedAlgebraicKnot,
    TorusCharacterOrbit,
    YanagidaTorusData,
    ckp_cable_levels,
    ckp_torus_knot_data,
    zero_surgery_twisted_alexander_torus_knot,
)
from gaknot.invariants.ckp import (
    CKPCableLevel,
    CKPLevelTerm,
    CKPRootMultiplicity,
    CKPTorusKnotData,
)
from gaknot.invariants.torus_twisted_blanchfield import local_valuation
from gaknot.invariants.twisted_alexander import twisted_alexander_torus_knot


@pytest.fixture(scope="module")
def two_by_five_data():
    r"""Return the nontrivial orbit ``(4,1)`` for ``T(2,5)``.

    The orbit sums to zero modulo five and is small enough that every matrix
    can be written explicitly.  Its two denominator factors cancel the roots
    ``zeta_5`` and ``zeta_5^4`` from ``1-t^5``; the remaining roots make the
    root-support bookkeeping independently visible.
    """
    orbit = TorusCharacterOrbit(
        p=2,
        q=5,
        generator_values=(QQ(1) / 5,),
        a_values=(4, 1),
    )
    return CKPTorusKnotData(orbit)


# ---------------------------------------------------------------------------
# Proposition 3.2: the conjugated metabelian representation
# ---------------------------------------------------------------------------

def test_ckp_shift_and_generator_images_are_exact(two_by_five_data):
    r"""Check the literal matrices displayed in Proposition 3.2.

    For ``p=2``, the shift matrix is ``[[0,1],[t,0]]``.  The first group
    generator maps to its fifth power, while the second maps to
    ``t*diag(zeta_5^4,zeta_5)`` in the cyclic order supplied by the character.
    This catches transposition, inverse-root, and orbit-order mistakes before
    they become hidden inside determinants.
    """
    data = two_by_five_data
    F = data.function_field
    t = data.t
    zeta = data.zeta

    expected_shift = matrix(F, [[0, 1], [t, 0]])
    expected_c2 = t * matrix.diagonal(F, [zeta ** 4, zeta])

    assert data.A == expected_shift
    assert data.c1_image == expected_shift ** 5
    assert data.c2_image == expected_c2


def test_ckp_matrices_satisfy_both_defining_relations(two_by_five_data):
    r"""Verify ``A^p=tI`` and the torus-group relation ``c1^p=c2^q``.

    The second equality uses both hypotheses on the orbit: every diagonal
    entry is a ``q``-th root, and the representation dimension/cover degree is
    ``p``.  A successful determinant calculation would not by itself prove
    that the supplied matrices define a representation, so this relation is
    tested directly.
    """
    data = two_by_five_data
    identity = identity_matrix(data.function_field, data.p)

    assert data.A ** data.p == data.t * identity
    assert data.c1_image ** data.p == data.c2_image ** data.q
    assert data.relation_holds is True


def test_ckp_frozen_record_does_not_expose_mutable_matrices(two_by_five_data):
    """Freeze both attribute assignment and individual Sage matrix entries."""
    data = two_by_five_data

    with pytest.raises(FrozenInstanceError):
        data.p = 7
    with pytest.raises(ValueError, match="immutable"):
        data.A[0, 0] = 1
    with pytest.raises(ValueError, match="immutable"):
        data.fox_numerator_matrix[0, 0] = 0


def test_ckp_rejects_an_object_that_is_not_a_character_orbit():
    """Fail at the dataclass boundary instead of reading arbitrary attributes."""
    with pytest.raises(TypeError, match="TorusCharacterOrbit"):
        CKPTorusKnotData((4, 1))


# ---------------------------------------------------------------------------
# Proposition 3.3 and Corollary 3.4: Fox determinants
# ---------------------------------------------------------------------------

def test_ckp_fox_numerator_is_matrix_geometric_sum(two_by_five_data):
    r"""Reconstruct the Fox derivative before taking its determinant.

    Differentiating ``c1^p c2^(-q)`` with respect to ``c1`` gives
    ``I+c1+...+c1^(p-1)``.  For this two-dimensional fixture that is simply
    ``I+c1``; the determinant must equal ``(1-t^q)^(p-1)`` exactly, not merely
    up to a Laurent unit.
    """
    data = two_by_five_data
    identity = identity_matrix(data.function_field, data.p)

    assert data.fox_numerator_matrix == identity + data.c1_image
    assert data.fox_numerator_determinant == (
        (1 - data.t ** data.q) ** (data.p - 1)
    )


def test_ckp_fox_denominator_is_det_c2_minus_identity(two_by_five_data):
    r"""Check equation (13) both as a determinant and a product of factors."""
    data = two_by_five_data
    identity = identity_matrix(data.function_field, data.p)
    expected_product = (
        (data.t * data.zeta ** 4 - 1)
        * (data.t * data.zeta - 1)
    )

    assert data.fox_denominator_determinant == (
        data.c2_image - identity
    ).det()
    assert data.fox_denominator_determinant == expected_product


def test_public_exterior_function_delegates_to_ckp_data():
    r"""Keep the historical API synchronized with Proposition 3.2 data.

    This test constructs the character through public Smith coordinates.  It
    therefore also checks the conversion to the deck orbit used by the CKP
    matrices, rather than relying only on the hand-built fixture above.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(3, 19)
    character = Character(
        BranchedCoverHomology(knot, 3),
        [[[QQ(1) / 19, QQ(0)]]],
    )
    data = ckp_torus_knot_data(knot, character)

    assert data.orbit.a_values == (18, 1, 0)
    assert twisted_alexander_torus_knot(knot, character) == (
        data.exterior_twisted_alexander
    )


def test_zero_surgery_formula_has_exact_sign_and_t_minus_one_factor():
    r"""Check the fixed representative in Corollary 3.4.

    Twisted Alexander invariants are normally defined up to units, but this
    API deliberately preserves the paper's displayed representative.  The
    factor ``(-1)^(p-1)`` is consequently tested even though it would be
    harmless at the level of an abstract order ideal.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    character = Character(
        BranchedCoverHomology(knot, 2),
        [[[QQ(1) / 5]]],
    )
    data = ckp_torus_knot_data(knot, character)

    expected = (
        (-1) ** (data.p - 1)
        * data.exterior_twisted_alexander
        / (data.t - 1)
    )
    assert data.zero_surgery_twisted_alexander == expected
    assert zero_surgery_twisted_alexander_torus_knot(
        knot,
        character,
    ) == expected


def test_ckp_and_yanagida_use_the_same_global_matrix_conventions():
    r"""Cross-check the two papers where their global formulas overlap.

    Yanagida writes the CKP shift, first-generator image, and diagonal image as
    ``C``, ``X=C^q``, and ``Y``.  His zero-surgery order uses factors written
    with opposite signs, whose total sign is precisely the CKP convention.
    Equality here prevents the two independently useful APIs from silently
    adopting transposed shifts or inverse character orbits.
    """
    orbit = TorusCharacterOrbit(
        p=3,
        q=4,
        generator_values=(QQ(0), QQ(1) / 4),
        a_values=(0, 1, 3),
    )
    ckp = CKPTorusKnotData(orbit)
    yanagida = YanagidaTorusData(3, 4, (0, 1, 3))

    assert ckp.A == yanagida.C
    assert ckp.c1_image == yanagida.X
    assert ckp.c2_image == yanagida.Y
    assert ckp.zero_surgery_twisted_alexander == (
        yanagida.zero_surgery_twisted_alexander_order
    )


# ---------------------------------------------------------------------------
# Exact cyclotomic root bookkeeping
# ---------------------------------------------------------------------------

def test_root_multiplicities_record_cancellation_and_orders(two_by_five_data):
    r"""Resolve the divisor of the exterior representative root by root.

    ``1-t^5`` has every fifth root once.  The orbit ``(4,1)`` removes roots at
    exponents one and four, leaving exponents zero, two, and three.  The root
    at exponent zero has order one; every nontrivial fifth root has order five.
    """
    roots = two_by_five_data.exterior_root_multiplicities

    assert tuple(root.exponent for root in roots) == (0, 1, 2, 3, 4)
    assert tuple(root.multiplicity for root in roots) == (1, 0, 1, 1, 0)
    assert tuple(root.root_order for root in roots) == (1, 5, 5, 5, 5)
    assert tuple(root.exponent for root in two_by_five_data.exterior_zero_support) == (
        0,
        2,
        3,
    )
    assert two_by_five_data.exterior_pole_support == ()


def test_zero_surgery_removes_one_additional_root_at_one(two_by_five_data):
    r"""Account for the extra ``t-1`` denominator without numerical roots."""
    roots = two_by_five_data.zero_surgery_root_multiplicities

    assert tuple(root.multiplicity for root in roots) == (0, 0, 1, 1, 0)
    assert tuple(
        root.exponent for root in two_by_five_data.zero_surgery_zero_support
    ) == (2, 3)
    assert two_by_five_data.zero_surgery_pole_support == ()


@pytest.mark.parametrize(
    "representative_name, multiplicities_name",
    [
        ("exterior_twisted_alexander", "exterior_root_multiplicities"),
        (
            "zero_surgery_twisted_alexander",
            "zero_surgery_root_multiplicities",
        ),
    ],
)
def test_combinatorial_root_table_equals_exact_local_valuations(
    two_by_five_data,
    representative_name,
    multiplicities_name,
):
    r"""Validate the root-counting shortcut against the rational functions.

    The public table is assembled by counting numerator and denominator
    factors, which avoids repeatedly factoring over a cyclotomic field.  For
    every fifth root in both fixed representatives, compare that count with
    an independent exact ``(t-zeta^k)`` valuation of the actual rational
    function.  This detects either a sign error in ``-a`` or an omitted
    surgery factor.
    """
    representative = getattr(two_by_five_data, representative_name)
    multiplicities = getattr(two_by_five_data, multiplicities_name)

    for root_record in multiplicities:
        root = two_by_five_data.zeta ** root_record.exponent
        assert local_valuation(representative, root) == (
            root_record.multiplicity
        )


def test_trivial_character_retains_poles_of_displayed_representatives():
    r"""Do not mislabel a rational representative as an honest polynomial.

    For the trivial orbit on ``T(3,5)``, all three denominator factors occur
    at ``t=1``.  The exterior numerator has multiplicity two there, giving one
    pole; zero surgery adds a second ``t-1`` denominator.  Recording these
    negative multiplicities is more informative than returning only zeros.
    """
    orbit = TorusCharacterOrbit(
        p=3,
        q=5,
        generator_values=(QQ(0), QQ(0)),
        a_values=(0, 0, 0),
    )
    data = CKPTorusKnotData(orbit)

    assert tuple(
        (root.exponent, root.multiplicity)
        for root in data.exterior_pole_support
    ) == ((0, -1),)
    assert tuple(
        (root.exponent, root.multiplicity)
        for root in data.zero_surgery_pole_support
    ) == ((0, -2),)


@pytest.mark.parametrize(
    "args, exception, message",
    [
        ((True, 0, 1), TypeError, "modulus must be an integer"),
        ((1, 0, 1), ValueError, "modulus must be greater than one"),
        ((5, "one", 1), TypeError, "exponent must be an integer"),
    ],
)
def test_root_multiplicity_validates_exact_integer_data(args, exception, message):
    """Reject ambiguous root labels before normalizing their exponents."""
    with pytest.raises(exception, match=message):
        CKPRootMultiplicity(*args)


# ---------------------------------------------------------------------------
# Section 5.1: s-levels and root-separation metadata
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ckp_four_summand_knot():
    r"""Return a concrete algebraically cancelling member of the CKP family.

    This is ``J(3,2,19,23)`` in the introductory four-term pattern.  The
    inequalities ``19,23 > 3*3*2`` also make both iterated torus knots
    algebraic knots, although the level calculation itself needs only their
    common first parameter.
    """
    return GeneralizedAlgebraicKnot([
        (1, [(3, 2), (3, 19)]),
        (-1, [(3, 19)]),
        (-1, [(3, 2), (3, 23)]),
        (1, [(3, 23)]),
    ])


def test_ckp_levels_reproduce_both_formal_cancellations(ckp_four_summand_knot):
    r"""Check the zero and first levels of ``J(3,2,19,23)``.

    Level zero contains ``T(3,19)-T(3,19)-T(3,23)+T(3,23)``.  Level one
    receives only the two iterated summands and is ``T(3,2)-T(3,2)``.  Higher
    levels are the unknot and are represented by the end of the returned
    tuple rather than an infinite sequence of empty records.
    """
    levels = ckp_cable_levels(ckp_four_summand_knot)

    assert len(levels) == 2
    assert levels[0].signed_multiplicities == ((19, 0), (23, 0))
    assert levels[1].signed_multiplicities == ((2, 0),)
    assert all(level.is_formally_zero for level in levels)
    assert ckp_four_summand_knot.ckp_cable_levels() == levels


def test_ckp_levels_preserve_source_layers_signs_and_substitution_powers(
    ckp_four_summand_knot,
):
    r"""Retain enough provenance to audit equation (18) term by term."""
    level_zero, level_one = ckp_four_summand_knot.ckp_cable_levels()

    assert tuple(
        (term.sign, term.q, term.source_component, term.source_layer)
        for term in level_zero.terms
    ) == (
        (1, 19, 0, 1),
        (-1, 19, 1, 0),
        (-1, 23, 2, 1),
        (1, 23, 3, 0),
    )
    assert tuple(
        (term.sign, term.q, term.source_component, term.source_layer)
        for term in level_one.terms
    ) == ((1, 2, 0, 0), (-1, 2, 2, 0))

    assert level_zero.substitution_power == 1
    assert level_one.substitution_power == 3
    assert level_zero.root_moduli == (57, 69)
    assert level_one.root_moduli == (18,)


def test_ckp_levels_reject_mixed_first_cabling_parameters():
    r"""Enforce the common-``p`` hypothesis of the Section 5 family."""
    mixed = GeneralizedAlgebraicKnot([
        (1, [(3, 2), (3, 19)]),
        (-1, [(2, 19)]),
    ])

    with pytest.raises(ValueError, match="same first parameter p"):
        ckp_cable_levels(mixed)


def test_ckp_levels_reject_non_knot_input():
    """Give an API-level diagnostic rather than an attribute error."""
    with pytest.raises(TypeError, match="GeneralizedAlgebraicKnot"):
        ckp_cable_levels([[(3, 2)]])


def test_public_level_record_checks_derived_multiplicities():
    r"""Prevent a manually constructed record from concealing a nonzero term."""
    term = CKPLevelTerm(
        sign=1,
        p=3,
        q=2,
        source_component=0,
        source_layer=0,
        root_modulus=6,
    )

    with pytest.raises(ValueError, match="signed collection"):
        CKPCableLevel(
            s=0,
            p=3,
            substitution_power=1,
            terms=(term,),
            signed_multiplicities=((2, 0),),
        )
