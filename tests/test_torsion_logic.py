"""Tests for torsion elements and characters supported on the torsion subgroup.

The homology implementation uses invariant factor ``0`` to denote a free
``Z`` summand and a positive factor ``m`` to denote ``Z/mZ``.  Consequently,
an element is torsion exactly when all coordinates belonging to zero factors
vanish.  The tests below check that rule at construction time, evaluation
time, and after arithmetic on homology elements.
"""

import pytest
from sage.all import QQ, Integer

from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.character import Character


# A Character object represents a character on the torsion subgroup.  Its
# nested input must therefore be zero wherever the homology decomposition has
# a free factor.  Valid values are flattened and normalized into [0, 1).
@pytest.mark.parametrize("knot_desc, N, char_values, should_fail, expected_values", [
    # 1. On Z + Z, the nonzero value 1/2 violates the required zero restriction.
    ([(1, [(2, 3)])], 6, [[[QQ(1) / 2, 0]]], True, None),
    # 2. Pure torsion-free: T(2,3), N=6. The zero character is valid.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], False, [QQ(0), QQ(0)]),
    # 3. Pure torsion: T(2,3), N=2. The input 4/3 is normalized to 1/3 in Q/Z.
    ([(1, [(2, 3)])], 2, [[[QQ(4) / 3]]], False, [QQ(1) / 3]),
    # 4. In Z/5Z + Z + Z, the exact value 1/10 on a free generator is invalid.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1) / 5]], [[QQ(1) / 10, 0]]], True, None),
    # 5. Mixed: the Z/5Z value is retained and the two free values remain zero.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(2) / 5]], [[0, 0]]], False, [QQ(2) / 5, QQ(0), QQ(0)]),
    # 6. In Z + Z + (Z/2Z)^4, the value 1 on a free generator is invalid.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 1], [0, 0, 0, 0]]], True, None),
    # 7. Nested satellite: free values stay zero and the Z/2Z values are flattened in layer order.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 0], [QQ(1) / 2, QQ(1) / 2, 0, 0]]], False,
     [QQ(0), QQ(0), QQ(1) / 2, QQ(1) / 2, QQ(0), QQ(0)]),
    # 8. Large factors: T(2,11), N=2. Factor [11].
    ([(1, [(2, 11)])], 2, [[[QQ(10) / 11]]], False, [QQ(10) / 11]),
    # 9. Concordance inverse: -T(2,3), N=6. Factors [0, 0].
    ([(-1, [(2, 3)])], 6, [[[QQ(1) / 3, 0]]], True, None),
    # 10. A nonzero value on the last of four free generators is invalid.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [[[0, 0]], [[0, QQ(1) / 10]]], True, None)
])
def test_character_torsion_free_restriction(
    knot_desc, N, char_values, should_fail, expected_values
):
    """Require character values to vanish on every free homology generator."""
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)

    if should_fail:
        # A nonzero free-coordinate value does not define one of the
        # torsion-supported characters represented by this class.
        with pytest.raises(ValueError, match="Characters must be zero on the torsion-free part"):
            Character(h1, char_values)
    else:
        # Successful construction must also flatten and normalize the values
        # in the same order as the homology invariant factors.
        char = Character(h1, char_values)
        assert char.values == expected_values


# Evaluation is a second line of defense: even a valid torsion-supported
# character may only be called on an element of the torsion subgroup.
@pytest.mark.parametrize("knot_desc, N, char_values, element_values, should_fail, expected_eval", [
    # 1. T(2,3), N=6. Pure free. el=[1,0] is not torsion, so evaluation is undefined.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], [1, 0], True, None),
    # 2. T(2,3), N=6. The zero element is torsion, and the only permitted character is zero.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], [0, 0], False, QQ(0)),
    # 3. T(2,3), N=2. In Z/3Z, [5]=[2], so a generator value of 1/3 gives 2/3.
    ([(1, [(2, 3)])], 2, [[[QQ(1) / 3]]], [5], False, QQ(2) / 3),
    # 4. The mixed group has factors [5,0,0]; the element uses only the Z/5Z generator.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1) / 5]], [[0, 0]]], [1, 0, 0], False, QQ(1) / 5),
    # 5. A nonzero coordinate in either free summand makes the element non-torsion.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1) / 5]], [[0, 0]]], [0, 1, 0], True, None),
    # 6. In the Z/5Z summand, [7]=[2], whose image under the character is 2/5.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1) / 5]], [[0, 0]]], [7, 0, 0], False, QQ(2) / 5),
    # 7. The first coordinate belongs to a free summand, so the element is not torsion.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 0], [0, 0, 0, 0]]], [1, 0, 0, 0, 0, 0], True, None),
    # 8. Four values of 1/2 sum to 2, which represents zero in Q/Z.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 0], [QQ(1) / 2, QQ(1) / 2, QQ(1) / 2, QQ(1) / 2]]], [0, 0, 1, 1, 1, 1], False, QQ(0)),
    # 9. On Z/3Z + Z/3Z, the two generator images add to 2/3.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 2, [[[QQ(1) / 3]], [[QQ(1) / 3]]], [1, 1], False, QQ(2) / 3),
    # 10. In a purely free group the zero element evaluates to zero under the zero character.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [[[0, 0]], [[0, 0]]], [0, 0, 0, 0], False, QQ(0))
])
def test_character_evaluation_torsion_only(
    knot_desc, N, char_values, element_values, should_fail, expected_eval
):
    """Evaluate torsion elements exactly and reject elements of infinite order."""
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    char = Character(h1, char_values)
    el = h1.element(element_values)

    if should_fail:
        with pytest.raises(ValueError, match="Character evaluation is only defined for torsion elements"):
            char(el)
    else:
        # Character values lie in Q/Z, represented canonically by rationals in
        # [0, 1), so exact equality is appropriate here.
        assert char(el) == expected_eval


# These cases isolate the element-level predicate.  Expected factors are part
# of each fixture so a homology-decomposition regression is distinguishable
# from a regression in ``BranchedCoverHomologyElement.is_torsion`` itself.
@pytest.mark.parametrize("knot_desc, N, expected_factors, el_values, expected_is_torsion", [
    # 1. In the free group Z + Z, the zero element is torsion.
    ([(1, [(2, 3)])], 6, [0, 0], [0, 0], True),
    # 2. A nonzero coordinate in Z + Z has infinite order.
    ([(1, [(2, 3)])], 6, [0, 0], [1, 0], False),
    # 3. Every element of Z/3Z is torsion, including its generator [1].
    ([(1, [(2, 3)])], 2, [3], [1], True),
    # 4. The other nonzero element [2] of Z/3Z is torsion as well.
    ([(1, [(2, 3)])], 2, [3], [2], True),
    # 5. In Z/5Z + Z + Z, an element supported on Z/5Z is torsion.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [5, 0, 0], [3, 0, 0], True),
    # 6. In the same group, a nonzero free coordinate gives infinite order.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [5, 0, 0], [0, 0, 1], False),
    # 7. This satellite element vanishes on both free generators.
    ([(1, [(2, 3), (2, 3)])], 6, [0, 0, 2, 2, 2, 2], [0, 0, 1, 0, 1, 0], True),
    # 8. Adding a nonzero first free coordinate makes it non-torsion.
    ([(1, [(2, 3), (2, 3)])], 6, [0, 0, 2, 2, 2, 2], [1, 0, 1, 0, 1, 0], False),
    # 9. The second coordinate is nonzero in the purely free connected sum.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [0, 0, 0, 0], [0, 1, 0, 0], False),
    # 10. The zero element remains torsion in a free group of rank four.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [0, 0, 0, 0], [0, 0, 0, 0], True)
])
def test_element_is_torsion(
    knot_desc, N, expected_factors, el_values, expected_is_torsion
):
    """Classify elements using their coordinates in the invariant-factor basis."""
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    el = h1.element(el_values)

    # The torsion decision is meaningful only relative to this decomposition:
    # a zero factor denotes a free Z summand, while positive factors denote
    # finite cyclic summands.
    assert h1.all_invariant_factors == [Integer(f) for f in expected_factors]
    assert el.is_torsion == expected_is_torsion


# The remaining tests exercise the same predicate after less routine inputs
# and group operations, where torsion status can differ from that of an input.
def test_negative_free_coordinate_is_not_torsion():
    # The sixth branched cover of the trefoil has H_1 = Z + Z.  The sign of a
    # nonzero free coordinate does not affect its infinite order.
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 6)

    assert h1.all_invariant_factors == [Integer(0), Integer(0)]
    assert not h1.element([-1, 0]).is_torsion


def test_multiplication_by_zero_produces_a_torsion_element():
    # The starting element has infinite order because its first free
    # coordinate is nonzero.  Multiplication by zero must construct the group
    # identity, which is torsion even when the ambient group is torsion-free.
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 6)
    element = h1.element([2, -3])

    zero_multiple = 0 * element

    assert not element.is_torsion
    assert zero_multiple.values == [Integer(0), Integer(0)]
    assert zero_multiple.is_torsion


def test_addition_can_cancel_the_free_part():
    # This connected sum has H_1 = Z/5Z + Z + Z.  Both operands below have
    # infinite order, but their opposite free coordinates cancel.  Their sum
    # remains nonzero in Z/5Z and is therefore a nonzero torsion element.
    knot = GeneralizedAlgebraicKnot(
        [(1, [(2, 5)]), (1, [(2, 3)])]
    )
    h1 = BranchedCoverHomology(knot, 6)
    left = h1.element([1, 2, 0])
    right = h1.element([1, -2, 0])

    result = left + right

    assert h1.all_invariant_factors == [Integer(5), Integer(0), Integer(0)]
    assert not left.is_torsion
    assert not right.is_torsion
    assert result.values == [Integer(2), Integer(0), Integer(0)]
    assert result.is_torsion
