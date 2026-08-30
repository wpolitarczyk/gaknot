r"""End-to-end tests for common-``p`` iterated-torus twisted signatures.

The lower-level test files verify the three mathematical ingredients
independently:

* ``test_twisted_alexander.py`` checks the Smith-basis character orbit;
* ``test_torus_twisted_signature.py`` checks Yanagida's local pattern jumps;
* ``test_metabelian_satellite.py`` checks the abstract Theorem 4.19 algebra.

The purpose of this file is different.  It verifies that a concrete
``GeneralizedAlgebraicKnot`` and ``Character`` are routed through those layers
with the correct conventions.  In particular, it makes the following easy-to-
confuse choices explicit:

1. knot cable sequences run from the innermost knot to the outermost pattern;
2. branched-cover homology layers run in the reverse, outer-to-inner order;
3. the character orbit is read only from the outer layer in the common-``p``
   ``p``-fold cover;
4. the same ordered orbit supplies Yanagida's ``b`` vector and the phase
   arguments in Theorem 4.19; and
5. the entire inner cable sequence contributes its *ordinary*
   Levine--Tristram signature because the outer winding equals the cover
   degree.

Coverage gaps are tested just as carefully as known jumps.  Yanagida's
formulas do not cover ``t=1`` and sometimes leave exceptional nontrivial
primary modules unresolved.  An end-to-end function must preserve those gaps,
not silently report a zero jump.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import QQ

from gaknot import (
    AveragedTwistedSignatureFunction,
    BranchedCoverHomology,
    Character,
    GeneralizedAlgebraicKnot,
    IteratedTorusMetabelianSignatureFunctionResult,
    IteratedTorusMetabelianSignatureResult,
    TorusCharacterOrbit,
    iterated_torus_metabelian_signature_function,
    iterated_torus_metabelian_signature_jumps,
)
from gaknot.invariants.LT_signature import LT_signature_iterated_torus_knot
from gaknot.invariants.metabelian_satellite import Theorem419SignatureResult
from gaknot.invariants.signature import SignatureFunction
from gaknot.invariants.torus_twisted_signature import YanagidaSignatureProfile


def _zero_character(knot, cover_degree):
    """Build the zero character without assuming a particular layer shape.

    Validation tests below deliberately use several knot and cover shapes.
    Deriving each layer's coordinate count from the decomposition keeps those
    tests focused on the end-to-end domain condition rather than on manually
    counting unrelated Smith generators.
    """
    homology = BranchedCoverHomology(knot, cover_degree)
    nested_values = []
    for component in homology.decomposition:
        component_values = []
        for layer in component["layers"]:
            coordinate_count = (
                layer["multiplicity"] * len(layer["base_factors"])
            )
            component_values.append([0] * coordinate_count)
        nested_values.append(component_values)
    return Character(homology, nested_values)


def _double_cable_nontrivial_character():
    r"""Return ``T(2,3;2,5)`` with the outer generator sent to ``1/5``.

    The double cover decomposes into an outer ``Z/5Z`` pattern contribution
    and two copies of the one-fold cover of the trefoil companion.  The latter
    covers are three-spheres, so their homology layers contain no character
    coordinates.  The nested character input is therefore ``[[[1/5], []]]``.
    """
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    homology = BranchedCoverHomology(knot, 2)
    character = Character(homology, [[[QQ(1) / 5], []]])
    return knot, homology, character


# ---------------------------------------------------------------------------
# A complete data-flow regression for T(2,3;2,5)
# ---------------------------------------------------------------------------

def test_common_two_cable_routes_outer_character_and_inner_signature():
    r"""Check every intermediate object in the supported two-layer formula.

    In the package's convention, ``[(2,3),(2,5)]`` means the ``(2,5)`` cable
    of the trefoil.  Smith-to-companion conversion sends the outer character
    value ``1/5`` to the ordered orbit ``(4,1)``.  Thus the two ordinary
    companion summands are evaluated at phases ``4/5`` and ``1/5``.  This test
    verifies those facts before checking the final addition, so a future
    failure will identify whether the problem is decomposition, orbit
    transport, or signature assembly.
    """
    knot, homology, character = _double_cable_nontrivial_character()

    # Confirm the structural premise used by the high-level function.  The
    # homology order is outer T(2,5) first and inner T(2,3) second, opposite to
    # the cable-sequence order stored by the knot.
    outer_layer, inner_layer = homology.decomposition[0]["layers"]
    assert outer_layer == {
        "cable_index": 1,
        "parameters": (2, 5),
        "effective_N": 2,
        "multiplicity": 1,
        "base_factors": [5],
    }
    assert inner_layer == {
        "cable_index": 0,
        "parameters": (2, 3),
        "effective_N": 1,
        "multiplicity": 2,
        "base_factors": [],
    }

    result = iterated_torus_metabelian_signature_jumps(knot, character)

    assert isinstance(result, IteratedTorusMetabelianSignatureResult)
    assert result.cable_sequence == ((2, 3), (2, 5))
    assert isinstance(result.orbit, TorusCharacterOrbit)
    assert result.orbit.generator_values == (QQ(1) / 5,)
    assert result.orbit.a_values == (4, 1)
    assert result.orbit.phase_arguments == (QQ(4) / 5, QQ(1) / 5)

    # Yanagida writes this same orbit as b.  No reversal or cyclic shift is
    # allowed between the shared orbit helper and the pattern matrices.
    assert isinstance(result.yanagida_profile, YanagidaSignatureProfile)
    assert result.yanagida_profile.data.m == 2
    assert result.yanagida_profile.data.n == 5
    assert result.yanagida_profile.data.b == (4, 1)

    # The companion is the full cable sequence below the outer layer, here
    # just the trefoil T(2,3).  It is not the outer T(2,5) pattern and it is not
    # a twisted lower-cover profile in this divisible-winding branch.
    expected_companion = LT_signature_iterated_torus_knot([(2, 3)])
    assert result.companion_signature == expected_companion
    assert result.companion_signature.jumps_counter == {
        QQ(1) / 6: -1,
        QQ(5) / 6: 1,
    }

    theorem_result = result.satellite_result
    assert isinstance(theorem_result, Theorem419SignatureResult)
    assert theorem_result.case == "ordinary_companion"
    assert theorem_result.cover_degree == 2
    assert theorem_result.winding == 2
    assert theorem_result.h == 2
    assert theorem_result.phase_arguments == (QQ(4) / 5, QQ(1) / 5)
    assert len(result.companion_summands) == 2

    # Pulling the trefoil jumps back by phase 4/5 moves 5/6 to 1/30 and 1/6
    # to 11/30.  Phase 1/5 gives the second pair at 19/30 and 29/30.  The
    # exponent is one because winding/cover_degree = 2/2.
    assert result.companion_summands[0].known_jumps == (
        (QQ(1) / 30, 1),
        (QQ(11) / 30, -1),
    )
    assert result.companion_summands[1].known_jumps == (
        (QQ(19) / 30, 1),
        (QQ(29) / 30, -1),
    )

    # Add those four companion jumps to Yanagida's generic pattern jumps at
    # 2/5 and 3/5.  The zero-module exceptional roots at 1/5 and 4/5 do not
    # create gaps; the only missing local computation is t=1.
    assert result.total_profile.known_jumps == (
        (QQ(1) / 30, 1),
        (QQ(11) / 30, -1),
        (QQ(2) / 5, -1),
        (QQ(3) / 5, 1),
        (QQ(19) / 30, 1),
        (QQ(29) / 30, -1),
    )
    assert result.unresolved_arguments == (QQ(0),)
    assert not result.is_complete
    with pytest.raises(NotImplementedError, match="argument 0"):
        result.total_profile.jump_at(0)


def test_end_to_end_result_is_an_immutable_diagnostic_record():
    """Prevent callers from replacing the orbit after profiles were computed."""
    knot, _, character = _double_cable_nontrivial_character()
    result = iterated_torus_metabelian_signature_jumps(knot, character)

    with pytest.raises(FrozenInstanceError):
        result.orbit = TorusCharacterOrbit(2, 5, (0,), (0, 0))


# ---------------------------------------------------------------------------
# From complete nontrivial jumps to the Casson--Gordon signature function
# ---------------------------------------------------------------------------

def test_common_two_cable_has_normalized_averaged_twisted_signature():
    r"""Integrate the end-to-end jumps using Theorem 4.14's normalization.

    The outer character has order five, so it is nontrivial and of prime-power
    order.  BCP-II, Theorem 4.14(a), therefore makes the metabelian
    Blanchfield form representable.  All six nontrivial-root jumps were
    checked in the detailed routing test above and sum to zero.  Hence the
    missing jump at ``t=1`` is also zero, and the averaged signature is
    canonically normalized to vanish there.

    Traversing the circle from zero, the successive half-jumps are

    ``+1, -1, -1, +1, +1, -1``

    at arguments ``1/30, 11/30, 2/5, 3/5, 19/30, 29/30``.  Each crossing
    changes the regular value by twice that number.  The assertions sample
    every constant arc and several discontinuity midpoints, making this a
    genuine function-level test rather than another comparison of counters.
    """
    knot, _, character = _double_cable_nontrivial_character()

    result = iterated_torus_metabelian_signature_function(knot, character)

    assert isinstance(
        result,
        IteratedTorusMetabelianSignatureFunctionResult,
    )
    assert result.character_order == 5
    assert result.cable_sequence == ((2, 3), (2, 5))
    assert result.orbit.a_values == (4, 1)
    assert isinstance(
        result.signature_function,
        AveragedTwistedSignatureFunction,
    )

    # The original calculation remains available for audit and still records
    # Yanagida's local coverage limitation.  The normalized object records
    # separately that representability, rather than a guessed local matrix,
    # supplied its root-one jump.
    assert result.jump_result.unresolved_arguments == (QQ(0),)
    assert result.signature_function.root_one_jump_inferred
    assert result.signature_function.root_one_jump == 0
    assert result.jump_profile.is_complete
    assert result.jump_profile.known_jumps == (
        (QQ(1) / 30, 1),
        (QQ(11) / 30, -1),
        (QQ(2) / 5, -1),
        (QQ(3) / 5, 1),
        (QQ(19) / 30, 1),
        (QQ(29) / 30, -1),
    )

    # Representable normalization and periodicity at t=1.
    assert result(0) == 0
    assert result(1) == 0

    # One regular sample from each of the seven complementary arcs.
    assert result(QQ(1) / 60) == 0
    assert result(QQ(1) / 10) == 2
    assert result(QQ(23) / 60) == 0
    assert result(QQ(1) / 2) == -2
    assert result(QQ(37) / 60) == 0
    assert result(QQ(4) / 5) == 2
    assert result(QQ(59) / 60) == 0

    # At a root the averaged value lies halfway between the adjacent regular
    # values.  The first and third crossings test positive and negative jumps.
    assert result(QQ(1) / 30) == 1
    assert result(QQ(2) / 5) == -1

    # Theorem 4.14(b): the Casson--Gordon signature difference is -sigma_av.
    assert result.casson_gordon_signature_difference_at(QQ(1) / 2) == 2


def test_signature_function_requires_a_nontrivial_character():
    r"""Do not invoke Theorem 4.14 for the order-one character.

    The jump-level calculation intentionally accepts the zero character, as
    tested below, because its satellite decomposition is still meaningful.
    The function-level normalization is stricter: the theorem used to prove
    representability explicitly assumes a nontrivial character.
    """
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    character = Character(BranchedCoverHomology(knot, 2), [[[0], []]])

    with pytest.raises(ValueError, match="nontrivial character.*order one"):
        iterated_torus_metabelian_signature_function(knot, character)


def test_signature_function_requires_prime_power_character_order():
    r"""Distinguish character order from the containing cyclic modulus.

    The double cover of ``T(2,15)`` has a ``Z/15Z`` summand, and sending its
    generator to ``1/15`` gives a genuinely order-fifteen character.  Although
    the jump formulas can be evaluated, fifteen is not a prime power, so the
    current proof of representability cannot be applied to normalize a
    Casson--Gordon twisted signature function.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 15)
    character = Character(
        BranchedCoverHomology(knot, 2),
        [[[QQ(1) / 15]]],
    )

    with pytest.raises(ValueError, match="prime-power-order.*order 15"):
        iterated_torus_metabelian_signature_function(knot, character)


def test_representability_cannot_fill_exceptional_nontrivial_roots():
    r"""Retain Yanagida coverage gaps even for a prime-power character.

    The order-four character on the outer ``T(3,4)`` layer satisfies the
    number-theoretic hypothesis of Theorem 4.14.  Representability determines
    the *sum* of all jumps, but the local contributions at ``1/4`` and ``3/4``
    remain individually unknown.  Since a signature function needs both
    discontinuities separately, the high-level call must stop rather than use
    the zero-total equation twice.
    """
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(3, 2), (3, 4)]
    )
    character = Character(
        BranchedCoverHomology(knot, 3),
        [[[QQ(1) / 4, 0], []]],
    )

    with pytest.raises(
        NotImplementedError,
        match=r"nontrivial arguments.*1/4.*3/4",
    ):
        iterated_torus_metabelian_signature_function(knot, character)


# ---------------------------------------------------------------------------
# Boundary cases and longer common-p sequences
# ---------------------------------------------------------------------------

def test_one_layer_torus_knot_uses_the_unknot_as_companion():
    r"""Treat an ordinary ``T(p,q)`` as a pattern applied to the unknot.

    Theorem 4.19 still produces one ordinary companion summand per orbit
    entry, but the unknot's Levine--Tristram signature is empty, so every such
    summand is the zero jump profile.  The total must therefore equal the
    Yanagida pattern profile rather than failing on an empty inner sequence.
    """
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    character = Character(
        BranchedCoverHomology(knot, 2),
        [[[QQ(1) / 5]]],
    )

    result = iterated_torus_metabelian_signature_jumps(knot, character)

    assert result.cable_sequence == ((2, 5),)
    assert result.companion_signature == SignatureFunction()
    assert len(result.companion_summands) == 2
    assert all(not summand.known_jumps for summand in result.companion_summands)
    assert result.total_profile.known_jumps == result.pattern_profile.known_jumps
    assert result.total_profile.unresolved == result.pattern_profile.unresolved


def test_three_layer_companion_contains_every_layer_below_the_pattern():
    r"""Do not discard deeper layers when removing the outermost pattern."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5), (2, 7)]
    )
    character = Character(
        BranchedCoverHomology(knot, 2),
        [[[QQ(1) / 7], [], []]],
    )

    result = iterated_torus_metabelian_signature_jumps(knot, character)

    assert result.orbit.a_values == (6, 1)
    assert result.companion_signature == LT_signature_iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    assert result.satellite_result.case == "ordinary_companion"
    assert len(result.companion_summands) == 2


def test_trivial_character_gives_zero_phases_but_keeps_all_summands():
    r"""The zero character simplifies phases, not the direct-sum multiplicity."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    character = Character(BranchedCoverHomology(knot, 2), [[[0], []]])

    result = iterated_torus_metabelian_signature_jumps(knot, character)

    assert result.orbit.a_values == (0, 0)
    assert result.orbit.phase_arguments == (0, 0)
    assert len(result.companion_summands) == 2
    assert (
        result.companion_summands[0].known_jumps
        == result.companion_summands[1].known_jumps
    )


def test_common_three_cable_preserves_positive_dimensional_exceptional_gaps():
    r"""Propagate unresolved Yanagida roots instead of guessing their jumps.

    For the outer pattern ``T(3,4)`` and orbit ``(3,1,0)``, the roots with
    indices 1 and 3 are exceptional and their local modules both have
    dimension one.  The high-level result must report gaps at arguments 1/4
    and 3/4, in addition to Yanagida's universal missing root at zero.  The
    ordinary trefoil companion contributes known jumps only and cannot resolve
    or cancel those unknown pattern summands.
    """
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(3, 2), (3, 4)]
    )
    character = Character(
        BranchedCoverHomology(knot, 3),
        [[[QQ(1) / 4, 0], []]],
    )

    result = iterated_torus_metabelian_signature_jumps(knot, character)

    assert result.orbit.a_values == (3, 1, 0)
    assert tuple(
        (exceptional.a, exceptional.module_dimension)
        for exceptional in result.yanagida_profile.exceptional_roots
    ) == ((1, 1), (3, 1))
    assert result.unresolved_arguments == (
        QQ(0),
        QQ(1) / 4,
        QQ(3) / 4,
    )
    for argument in result.unresolved_arguments:
        with pytest.raises(NotImplementedError):
            result.total_profile.jump_at(argument)


def test_equivalent_knot_instance_is_accepted_structurally():
    """Require matching descriptions, not Python object identity."""
    target = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    equivalent = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    character = Character(
        BranchedCoverHomology(equivalent, 2),
        [[[QQ(1) / 5], []]],
    )

    result = iterated_torus_metabelian_signature_jumps(target, character)

    assert result.cable_sequence == ((2, 3), (2, 5))
    assert result.orbit.a_values == (4, 1)


# ---------------------------------------------------------------------------
# Explicit boundary of the current high-level implementation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_knot",
    [None, "T(2,3;2,5)"],
    ids=["none", "string"],
)
def test_end_to_end_formula_rejects_non_knot_objects(invalid_knot):
    """Fail at the public type boundary before reading a knot description."""
    valid_knot, _, valid_character = _double_cable_nontrivial_character()
    del valid_knot

    with pytest.raises(TypeError, match="GeneralizedAlgebraicKnot"):
        iterated_torus_metabelian_signature_jumps(
            invalid_knot,
            valid_character,
        )


def test_end_to_end_formula_rejects_non_character_objects():
    """Do not interpret an arbitrary nested list as a validated character."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )

    with pytest.raises(TypeError, match="character must be a Character"):
        iterated_torus_metabelian_signature_jumps(knot, [[[QQ(1) / 5], []]])


def test_end_to_end_formula_rejects_nonconstant_p_sequence():
    r"""Keep the first public milestone restricted to common-``p`` cables."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (3, 4)]
    )
    character = _zero_character(knot, 3)

    with pytest.raises(NotImplementedError, match="same p parameter"):
        iterated_torus_metabelian_signature_jumps(knot, character)


def test_end_to_end_formula_rejects_wrong_cover_degree():
    r"""The cover must equal the common winding, even for the zero character."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    character = _zero_character(knot, 3)

    with pytest.raises(ValueError, match="2-fold cover"):
        iterated_torus_metabelian_signature_jumps(knot, character)


def test_end_to_end_formula_rejects_character_from_another_knot():
    """Do not reuse Smith coordinates belonging to a different outer pattern."""
    target = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    other = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 7)]
    )
    character = _zero_character(other, 2)

    with pytest.raises(ValueError, match="homology of the supplied knot"):
        iterated_torus_metabelian_signature_jumps(target, character)


@pytest.mark.parametrize("sign", [-1], ids=["negative"])
def test_end_to_end_formula_rejects_negative_iterated_knot(sign):
    """Record that orientation reversal has not yet been wired end to end."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)],
        sign=sign,
    )
    character = _zero_character(knot, 2)

    with pytest.raises(NotImplementedError, match="one positive"):
        iterated_torus_metabelian_signature_jumps(knot, character)


def test_end_to_end_formula_rejects_connected_sum():
    """Leave signed direct-sum character bookkeeping to the next milestone."""
    first = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    second = GeneralizedAlgebraicKnot.torus_knot(2, 7)
    knot_sum = first + second
    character = _zero_character(knot_sum, 2)

    with pytest.raises(NotImplementedError, match="one positive"):
        iterated_torus_metabelian_signature_jumps(knot_sum, character)
