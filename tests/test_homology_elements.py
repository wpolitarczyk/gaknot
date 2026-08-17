"""Tests for elements of branched-cover first homology groups.

``BranchedCoverHomologyElement`` represents an element by integer coordinates
paired, in structural order, with ``homology.all_invariant_factors``.  For a
factor ``m > 1``, the corresponding coordinate belongs to ``Z/mZ`` and is
stored as its canonical residue in ``0, ..., m-1``.  A factor ``0`` denotes a
free copy of ``Z``; its coordinate is an ordinary integer and is not reduced.

Callers may provide coordinates in either of two forms:

* a flat list following components, outer-to-inner layers, repeated copies,
  and finally factors within one copy; or
* a nested list with shape ``[component][layer][coordinate]``.

Multiplicity does not add another nesting level.  If a layer contains base
factors ``[m_1,m_2]`` and occurs twice, its one layer list contains four values
in copy-major order: ``[copy1_m1, copy1_m2, copy2_m1, copy2_m2]``.  Empty
layers remain present as explicit empty lists even though they add no flat
coordinates.

All arithmetic constructs a new element through the same normalization path.
Thus finite coordinates are reduced after addition, subtraction, negation,
and scalar multiplication, whereas free coordinates retain ordinary integer
arithmetic.  Character evaluation uses the matching structural coordinate
order, computes an exact dot product in ``QQ``, and reduces the result modulo
``Z`` to its representative in ``[0,1)``.
"""

import pytest
from sage.all import QQ, ZZ, Integer
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.H1_branched_cover import BranchedCoverHomologyElement
from gaknot.invariants.character import Character


# ---------------------------------------------------------------------------
# Construction, flattening, and coordinate normalization
# ---------------------------------------------------------------------------

# The table deliberately mixes flat and nested input.  Expected values are
# always flat because that is the element's normalized internal/public form.
# Together the rows cover finite reduction, unreduced free coordinates,
# component boundaries, satellite-layer boundaries, and empty layers.
@pytest.mark.parametrize("knot_desc, N, input_values, expected_values", [
    # A single Z/3Z coordinate reduces 4 to its canonical residue 1.
    ([(1, [(2, 3)])], 2, [4], [1]),
    # Flat input crosses a component boundary: 4 mod 3 and 7 mod 5.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [4, 7], [1, 2]),
    # Flat structural order for this satellite is outer Z/5 first, followed by
    # two inner Z/3 copies; each coordinate is reduced by its paired factor.
    ([(1, [(2, 3), (2, 5)])], 4, [6, 4, 2], [1, 1, 2]),
    # Nested input separates the two connected-sum components explicitly.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 2, [[[1]], [[2]]], [1, 2]),
    # One component contains outer coordinates [1] and a flat inner-layer list
    # [4,5] for its two repeated copies; flattening also reduces them mod 3.
    ([(1, [(2, 3), (2, 5)])], 4, [[[1], [4, 5]]], [1, 1, 2]),
    # The triple trefoil cover has two distinct Smith generators of order two.
    # This is one layer of multiplicity one, not satellite-copy multiplicity.
    ([(1, [(2, 3)])], 3, [5, 6], [1, 0]),
    # Zero factors mean Z summands, so positive and negative inputs are retained
    # exactly instead of being passed to a modulus operation.
    ([(1, [(2, 3)])], 6, [5, -2], [5, -2]),
    # The five-fold cover has four distinct order-two Smith generators in one
    # ordinary torus-knot layer; again, layer multiplicity itself remains one.
    ([(1, [(2, 5)])], 5, [1, 2, 3, 4], [1, 0, 1, 0]),
    # Three components flatten from left to right as moduli 3, 5, and 7.
    ([(1, [(2, 3)]), (1, [(2, 5)]), (1, [(2, 7)])], 2, [1, 1, 1], [1, 1, 1]),
    # Three satellite layers must all appear in nested input.  The innermost
    # layer contributes no factors and is therefore represented by ``[]``.
    ([(1, [(2, 3), (2, 5), (2, 7)])], 4, [[[8], [6, 11], []]], [1, 1, 1])
])
def test_homology_element_creation(knot_desc, N, input_values, expected_values):
    """Flat and nested descriptions produce the same normalized coordinates."""
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    # The group factory delegates shape validation, flattening, and reduction
    # to ``BranchedCoverHomologyElement``.
    el = h1.element(input_values)
    # Use Sage integers in the expectation to assert both values and public
    # coordinate type.
    assert el.values == [Integer(v) for v in expected_values]


def test_homology_element_values_are_defensively_copied():
    """The public coordinate list cannot mutate the normalized element."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 2)
    element = h1.element([1])

    # Change an existing coordinate and even alter the returned list's length.
    # If ``values`` exposed internal storage, this would bypass both the mod-3
    # normalization and the one-generator shape invariant.
    exported_values = element.values
    exported_values[0] = 2
    exported_values.append(0)

    # A second access must produce the original, still-normalized coordinate.
    assert element.values == [1]


# ---------------------------------------------------------------------------
# Coordinatewise group arithmetic
# ---------------------------------------------------------------------------

# Every expected finite coordinate is already reduced modulo its structural
# factor.  Free-factor rows intentionally retain negative and large integers.
# The fixtures also cover several generator counts so a zip/order regression
# cannot hide behind a one-coordinate example.
@pytest.mark.parametrize("knot_desc, N, v1, v2, scalar, expected_sum, expected_diff, expected_neg, expected_mul", [
    # In Z/3, 1+2=0, 1-2=-1=2, and -1=2.
    ([(1, [(2, 3)])], 2, [1], [2], 2, [0], [2], [2], [2]),
    # Connected-sum coordinates use their own moduli independently: 3 and 5.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [1, 2], [1, 4], 3, [2, 1], [0, 3], [2, 3], [0, 1]),
    # Two order-two generators check componentwise reduction and self-inverses.
    ([(1, [(2, 3)])], 3, [1, 0], [1, 1], 2, [0, 1], [0, 1], [1, 0], [0, 0]),
    # Free coordinates use ordinary Z arithmetic with no modular reduction.
    ([(1, [(2, 3)])], 6, [1, 2], [3, 4], 5, [4, 6], [-2, -2], [-1, -2], [5, 10]),
    # A satellite mixes one mod-5 coordinate with two mod-3 coordinates.
    ([(1, [(2, 3), (2, 5)])], 4, [1, 1, 1], [4, 2, 2], 2, [0, 0, 0], [2, 2, 2], [4, 2, 2], [2, 2, 2]),
    # Modulus four distinguishes negation from an order-two self-inverse.
    ([(1, [(3, 4)])], 3, [1, 1], [2, 3], 2, [3, 0], [3, 2], [3, 3], [2, 2]),
    # Four order-two generators exercise longer coordinatewise operations.
    ([(1, [(2, 5)])], 5, [1, 0, 1, 0], [1, 1, 0, 0], 3, [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]),
    # Modulus seven supplies nontrivial wraparound for every operation.
    ([(1, [(2, 7)])], 2, [2], [6], 4, [1], [3], [5], [1]),
    # Free coordinates from two components must remain in component order.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 6, [1, 1, 1, 1], [1, 2, 3, 4], 2, [2, 3, 4, 5], [0, -1, -2, -3], [-1, -1, -1, -1], [2, 2, 2, 2]),
    # Deep cabling can still flatten to one surviving cyclic coordinate.
    ([(1, [(2, 3), (2, 5), (2, 7)])], 2, [3], [5], 3, [1], [5], [4], [2])
])
def test_homology_element_arithmetic(knot_desc, N, v1, v2, scalar, expected_sum, expected_diff, expected_neg, expected_mul):
    """All group operations normalize finite and free coordinates correctly."""
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    el1 = h1.element(v1)
    el2 = h1.element(v2)
    
    # Binary operations require the shared parent group and preserve its
    # structural coordinate order.
    assert (el1 + el2).values == [Integer(v) for v in expected_sum]
    assert (el1 - el2).values == [Integer(v) for v in expected_diff]
    # Unary negation and both scalar spellings re-enter the same normalization
    # path; left and right scalar multiplication must therefore agree.
    assert (-el1).values == [Integer(v) for v in expected_neg]
    assert (el1 * scalar).values == [Integer(v) for v in expected_mul]
    assert (scalar * el1).values == [Integer(v) for v in expected_mul]


# ---------------------------------------------------------------------------
# Character evaluation on homology elements
# ---------------------------------------------------------------------------

# Character values and element coordinates share structural order.  Evaluation
# is their exact dot product in QQ, reduced modulo one.  The table covers one
# and several generators, connected sums, satellite layers, cancellation in
# Q/Z, and the required zero character on free factors.
@pytest.mark.parametrize("knot_desc, N, char_values, element_values, expected_eval", [
    # The generator of Z/3 maps to 1/3.
    ([(1, [(2, 3)])], 2, [[[QQ(1)/3]]], [1], QQ(1)/3),
    # (2/3)*2 = 4/3 represents 1/3 modulo Z.
    ([(1, [(2, 3)])], 2, [[[QQ(2)/3]]], [2], QQ(1)/3),
    # Connected-sum coordinates add: 1/3 + 1/5 = 8/15.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [[[QQ(1)/3]], [[QQ(1)/5]]], [1, 1], QQ(8)/15),
    # Two half-values sum to the zero class in Q/Z.
    ([(1, [(2, 3)])], 3, [[[QQ(1)/2, QQ(1)/2]]], [1, 1], 0),
    # A torsion-supported Character is forced to zero on free factors, and the
    # zero element is the only torsion element of this purely free group.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], [0, 0], 0),
    # Satellite layers contribute 2/5 + 1/3 + 1/3 = 16/15, hence 1/15 mod Z.
    ([(1, [(2, 3), (2, 5)])], 4, [[[QQ(2)/5], [QQ(1)/3, QQ(1)/3]]], [1, 1, 1], QQ(2)/5 + QQ(2)/3 - 1),
    # Quarter and three-quarter values cancel to one, the zero class.
    ([(1, [(3, 4)])], 3, [[[QQ(1)/4, QQ(3)/4]]], [1, 1], 0),
    # Only two of four order-two character coordinates are nonzero; they cancel.
    ([(1, [(2, 5)])], 5, [[[QQ(1)/2, 0, QQ(1)/2, 0]]], [1, 1, 1, 1], 0),
    # Values on separate connected-sum components still add in Q/Z.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 2, [[[QQ(1)/3]], [[QQ(2)/3]]], [1, 1], 0),
    # The innermost empty layer is structurally present but adds no term;
    # 1/7 + 1/5 + 2/5 = 26/35 is already in [0,1).
    ([(1, [(2, 3), (2, 5), (2, 7)])], 4, [[[QQ(1)/7], [QQ(1)/5, QQ(2)/5], []]], [1, 1, 1], QQ(1)/7 + QQ(3)/5 - 0),
])
def test_character_evaluation(knot_desc, N, char_values, element_values, expected_eval):
    """Characters pair with matching element coordinates and reduce modulo one."""
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    char = Character(h1, char_values)
    el = h1.element(element_values)
    # Exact Sage rationals make the Q/Z comparison independent of floating
    # approximation and expose any missing final reduction modulo one.
    assert char(el) == expected_eval


# ---------------------------------------------------------------------------
# Nested input for iterated torus knots
# ---------------------------------------------------------------------------

# Cable descriptions run from the innermost/base knot to the outermost pattern,
# whereas homology layers and nested element values run outer-to-inner.  At a
# layer with winding p and effective degree N, d=gcd(N,p); the next inner layer
# uses degree N/d and appears in d times as many copies.  Each row checks both
# the resulting structural factor order and flattening of its nested values.
@pytest.mark.parametrize("cable_desc, N, expected_factors, nested_values, expected_flat", [
    # Outer T(2,5) contributes Z/5.  Since gcd(2,2)=2, the inner degree is one
    # and both copies of its empty factor list flatten to no coordinates.
    ([(2, 3), (2, 5)], 2, [5], [[[1], []]], [1]),
    # At N=4 the inner degree becomes two; two copies of Z/3 follow outer Z/5.
    ([(2, 3), (2, 5)], 4, [5, 3, 3], [[[1], [1, 2]]], [1, 1, 2]),
    # Reversing the cable sequence makes T(2,3) the sole nontrivial outer layer.
    ([(2, 5), (2, 3)], 2, [3], [[[1], []]], [1]),
    # At N=6, outer Z/5 is followed by two copies of the inner triple-cover
    # factors [2,2], producing four consecutive order-two coordinates.
    ([(2, 3), (2, 5)], 6, [5, 2, 2, 2, 2], [[[1], [1, 1, 1, 1]]], [1, 1, 1, 1, 1]),
    # Repeating the trefoil pattern at degree two again leaves inner degree one.
    ([(2, 3), (2, 3)], 2, [3], [[[1], []]], [1]),
    # At degree four, outer Z/3 is followed by two inner Z/3 copies.
    ([(2, 3), (2, 3)], 4, [3, 3, 3], [[[1], [1, 2]]], [1, 1, 2]),
    # The outer six-fold trefoil cover supplies two free coordinates.  Two
    # copies of the inner triple-cover factors [2,2] follow them unchanged.
    ([(2, 3), (2, 3)], 6, [0, 0, 2, 2, 2, 2], [[[5, 6], [1, 1, 1, 1]]], [5, 6, 1, 1, 1, 1]),
    # Three layers at degree two retain only outer Z/7; both inner layer lists
    # remain present as empty structural placeholders.
    ([(2, 3), (2, 5), (2, 7)], 2, [7], [[[1], [], []]], [1]),
    # At degree four, outer Z/7 is followed by two middle Z/5 copies; the
    # innermost layer has effective degree one and contributes nothing.
    ([(2, 3), (2, 5), (2, 7)], 4, [7, 5, 5], [[[1], [2, 3], []]], [1, 2, 3]),
    # A third repeated layer demonstrates accumulated multiplicity: the middle
    # occurs twice and the empty innermost layer conceptually occurs four times.
    ([(2, 3), (2, 3), (2, 3)], 4, [3, 3, 3], [[[1], [1, 1], []]], [1, 1, 1])
])
def test_iterated_knot_structure(cable_desc, N, expected_factors, nested_values, expected_flat):
    """Nested satellite values flatten in outer-to-inner structural order."""
    knot = GeneralizedAlgebraicKnot([(1, cable_desc)])
    h1 = BranchedCoverHomology(knot, N)
    # Check the parent factor order first so a later coordinate mismatch is not
    # mistakenly attributed to the element flattener.
    assert h1.all_invariant_factors == [Integer(f) for f in expected_factors]
    el = h1.element(nested_values)
    assert el.values == [Integer(v) for v in expected_flat]


# ---------------------------------------------------------------------------
# Nested input across connected-sum components
# ---------------------------------------------------------------------------

# Connected sums add the outermost nesting level.  Component order follows the
# knot description, while each component independently retains its own
# outer-to-inner layers and multiplicities.  Empty homology still requires an
# explicit empty layer list so later components cannot shift position.
@pytest.mark.parametrize("sum_desc, N, expected_factors, nested_values, expected_flat", [
    # Two identical cyclic components retain distinct coordinates.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 2, [3, 3], [[[1]], [[2]]], [1, 2]),
    # Different component moduli flatten left-to-right as 3 then 5.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [3, 5], [[[1]], [[2]]], [1, 2]),
    # Three components verify that flattening is not specialized to pairs.
    ([(1, [(2, 3)]), (1, [(2, 3)]), (1, [(2, 3)])], 2, [3, 3, 3], [[[1]], [[1]], [[1]]], [1, 1, 1]),
    # A two-layer first component is completely flattened before the ordinary
    # trefoil component contributes its final mod-3 coordinate.
    ([(1, [(2, 3), (2, 5)]), (1, [(2, 3)])], 4, [5, 3, 3, 3], [[[1], [2, 2]], [[1]]], [1, 2, 2, 1]),
    # The triple trefoil cover contributes [2,2], whereas the triple cover of
    # T(2,5) is trivial.  Indeed, evaluating its Alexander polynomial at the
    # two primitive cube roots gives conjugate values whose product is one, so
    # the resultant has absolute value one and there are no nontrivial Smith
    # factors.  The second component is therefore represented by ``[[]]``.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 3, [2, 2], [[[1, 1]], [[]]], [1, 1]),
    # Two triple trefoil covers concatenate their two generators apiece.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 3, [2, 2, 2, 2], [[[1, 1]], [[1, 1]]], [1, 1, 1, 1]),
    # Free coordinates from separate components remain ordinary integers and
    # preserve their component boundary during flattening.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 6, [0, 0, 0, 0], [[[1, 2]], [[3, 4]]], [1, 2, 3, 4]),
    # Each five-fold T(2,5) cover contributes four order-two coordinates.
    ([(1, [(2, 5)]), (1, [(2, 5)])], 5, [2,2,2,2, 2,2,2,2], [[[1,0,1,0]], [[0,1,0,1]]], [1,0,1,0, 0,1,0,1]),
    # Heterogeneous multi-generator components retain factors [4,4] then [2,2].
    ([(1, [(3, 4)]), (1, [(2, 3)])], 3, [4, 4, 2, 2], [[[1, 1]], [[1, 1]]], [1, 1, 1, 1]),
    # Two satellites contribute only their respective outer factors; each
    # empty inner layer remains visible in its component's nested input.
    ([(1, [(2, 3), (2, 5)]), (1, [(2, 5), (2, 3)])], 2, [5, 3], [[[1], []], [[1], []]], [1, 1])
])
def test_connected_sum_structure(sum_desc, N, expected_factors, nested_values, expected_flat):
    """Nested component values flatten without losing component boundaries."""
    knot = GeneralizedAlgebraicKnot(sum_desc)
    h1 = BranchedCoverHomology(knot, N)
    # As in the satellite test, establish the exact structural moduli before
    # attributing the expected flat values to the element constructor.
    assert h1.all_invariant_factors == [Integer(f) for f in expected_factors]
    el = h1.element(nested_values)
    assert el.values == [Integer(v) for v in expected_flat]
