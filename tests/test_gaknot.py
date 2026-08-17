"""Tests for the core generalized-algebraic-knot data model.

``GeneralizedAlgebraicKnot`` stores a signed connected sum of iterated positive
torus knots.  A description has shape::

    [(sign, [(p_1,q_1), (p_2,q_2), ...]), ...]

Each outer pair is one connected-sum component.  The sign is ``+1`` for the
listed summand and ``-1`` for its concordance inverse.  Within a component,
the first cable pair is the innermost/base torus knot and every subsequent pair
is a new outer cabling pattern.

The representation intentionally preserves structure rather than simplifying
in the concordance group.  Addition concatenates component descriptions,
negation flips their signs, and subtraction combines those two operations.
Thus ``K # -K`` remains a two-component description even though it represents
a slice combination.  This stable ordering is consumed by homology,
characters, and signature calculations elsewhere in the package.

These tests cover four related contracts:

* structural validation and defensive ownership of descriptions;
* connected-sum, concordance, container, and classification operations;
* the Alexander polynomial under cabling, mirroring, and connected sum;
* convenience constructors and the one-summand cabling operation.
"""

import pytest
from sage.all import PolynomialRing, ZZ
from gaknot import GeneralizedAlgebraicKnot


def _normalized_description(desc):
    """Convert accepted tuple/list cable containers to the public list form.

    The constructor accepts both lists and tuples but normalizes cable
    sequences to owned lists.  Algebraic-operation tests use this helper when
    deriving expected descriptions from their raw parametrized inputs.
    """
    return [(sign, list(knot_desc)) for sign, knot_desc in desc]


# ---------------------------------------------------------------------------
# Construction, normalization, and validation
# ---------------------------------------------------------------------------

# The string table moves from one ordinary torus knot through signed iterated
# knots and multi-component connected sums.  Its purpose is not merely visual:
# it verifies that semicolons preserve inner-to-outer cabling order, ``#``
# preserves component order, and a minus sign applies to an entire summand.
@pytest.mark.parametrize("desc, expected_str", [
    # One positive and one negative ordinary torus knot.
    ([(1, [(2, 3)])], "T(2,3)"),
    ([(-1, [(2, 3)])], "-T(2,3)"),
    # The second pair is an outer cable of the first, not a second summand.
    ([(1, [(2, 3), (2, 5)])], "T(2,3; 2,5)"),
    ([(-1, [(2, 3), (2, 5)])], "-T(2,3; 2,5)"),
    # Separate outer entries become connected-sum components.
    ([(1, [(2, 3)]), (1, [(3, 4)])], "T(2,3) # T(3,4)"),
    ([(1, [(2, 3)]), (-1, [(3, 4)])], "T(2,3) # -T(3,4)"),
    # Mixed ordinary/iterated and signed/unsigned summands retain their order.
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3)])], "T(2,3; 6,5) # -T(2,3)"),
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3), (6, 7)])], "T(2,3; 6,5) # -T(2,3; 6,7)"),
    # Longer examples ensure formatting is not specialized to two entries.
    ([(1, [(2, 3)]), (1, [(3, 5)]), (1, [(5, 7)])], "T(2,3) # T(3,5) # T(5,7)"),
    ([(1, [(2, 3), (6, 5), (30, 7)])], "T(2,3; 6,5; 30,7)")
])
def test_gaknot_basic_functionality(desc, expected_str):
    """Construction preserves normalized data and renders its hierarchy."""
    knot = GeneralizedAlgebraicKnot(desc)
    # The readable form exposes sign, component, and cabling boundaries.
    assert str(knot) == expected_str
    # All rows already use lists, so normalization is equal to the input here;
    # tuple normalization is isolated below.
    assert knot.description == desc


# Each invalid row targets one invariant of the nested representation.  Error
# messages include structural indices so failures in long descriptions remain
# diagnosable rather than reporting only a generic malformed-input error.
@pytest.mark.parametrize("desc, error_type, match", [
    # The outer connected-sum description must be an explicit list or tuple.
    ("not a list", TypeError, "must be a list or tuple"),
    # Every component separates its sign from its cable sequence.
    ([(1, [(2, 3)]), (1,)], ValueError, "must be a pair"),
    # Only a summand and its concordance inverse are representable.
    ([(2, [(2, 3)])], ValueError, "Sign at index 0 must be 1 or -1"),
    # The per-summand cable sequence has the same list/tuple contract.
    ([(1, "not a list")], TypeError, "must be a list or tuple"),
    # Each cabling stage requires exactly the two parameters p and q.
    ([(1, [(2, 3, 4)])], ValueError, r"must be a pair \(p, q\)"),
    # Fractional cable parameters do not define the supported torus knots.
    ([(1, [(2.5, 3)])], TypeError, "must be integers"),
    # Both entries must be strictly greater than one in the positive model.
    ([(1, [(1, 3)])], ValueError, "must be > 1"),
    ([(1, [(3, 0)])], ValueError, "must be > 1"),
    # Coprime parameters produce a torus knot rather than a torus link.
    ([(1, [(2, 4)])], ValueError, "relatively prime"),
    # Validation applies to later outer cable patterns as well as the base.
    ([(1, [(2, 3), (6, 9)])], ValueError, "relatively prime")
])
def test_gaknot_validation_parametric(desc, error_type, match):
    """Malformed descriptions fail at the relevant structural invariant."""
    with pytest.raises(error_type, match=match):
        GeneralizedAlgebraicKnot(desc)


def test_empty_cable_sequence_rejected():
    """A connected-sum component must contain at least one torus-knot pair."""
    # An empty sequence would be neither an ordinary nor an iterated torus knot
    # and would make the component's invariants undefined.
    with pytest.raises(ValueError, match="at least one cabling pair"):
        GeneralizedAlgebraicKnot([(1, [])])


def test_tuple_descriptions_support_operations():
    """Tuple input is normalized once and remains usable by later operations."""
    # Exercise tuples at all three accepted container levels: connected-sum
    # description, component pair, and cable sequence/pair.
    tuple_knot = GeneralizedAlgebraicKnot(((1, ((2, 3),)),))
    list_knot = GeneralizedAlgebraicKnot([(1, [(2, 5)])])

    # Addition exports the normalized list representation rather than leaking
    # the tuple spelling used during construction.
    assert (tuple_knot + list_knot).description == [
        (1, [(2, 3)]),
        (1, [(2, 5)]),
    ]
    # Cabling likewise appends to the owned, normalized cable list.
    assert tuple_knot.cable(2, 5).description == [
        (1, [(2, 3), (2, 5)]),
    ]


def test_description_is_defensively_copied():
    """Neither constructor input nor an exported description aliases storage."""
    desc = [(1, [(2, 3)])]
    knot = GeneralizedAlgebraicKnot(desc)

    # First mutate the original nested list after construction.  If the knot
    # retained that list, its base torus knot would silently change to T(2,5).
    desc[0][1][0] = (2, 5)
    assert knot.description == [(1, [(2, 3)])]

    # Then mutate the list returned by the property.  A fresh nested copy must
    # make this equally harmless to the stored description.
    exported_description = knot.description
    exported_description[0][1][0] = (2, 7)
    assert knot.description == [(1, [(2, 3)])]


# ---------------------------------------------------------------------------
# Connected sum, concordance inverse, and subtraction
# ---------------------------------------------------------------------------

# The table mixes ordinary and iterated knots, positive and negative signs,
# one- and multi-component operands, repeated knots, and inverse pairs.  The
# expected strings are deliberately structural: these operations do not sort
# components or cancel ``K # -K``.
@pytest.mark.parametrize("desc1, desc2, expected_sum, expected_neg1, expected_diff", [
    # Basic addition, negation, and subtraction on positive torus knots.
    ([(1, [(2, 3)])], [(1, [(3, 4)])], "T(2,3) # T(3,4)", "-T(2,3)", "T(2,3) # -T(3,4)"),
    # Ordinary and iterated descriptions use the same component operations.
    ([(1, [(2, 5)])], [(1, [(2, 3), (6, 5)])], "T(2,5) # T(2,3; 6,5)", "-T(2,5)", "T(2,5) # -T(2,3; 6,5)"),
    # Negating an already negative component makes it positive.
    ([(-1, [(2, 3)])], [(1, [(3, 4)])], "-T(2,3) # T(3,4)", "T(2,3)", "-T(2,3) # -T(3,4)"),
    # An inverse pair remains visible; no concordance simplification occurs.
    ([(1, [(2, 3), (6, 5)])], [(-1, [(2, 3), (6, 5)])], "T(2,3; 6,5) # -T(2,3; 6,5)", "-T(2,3; 6,5)", "T(2,3; 6,5) # T(2,3; 6,5)"),
    # Multi-component left and right operands verify concatenation order.
    ([(1, [(2, 3)]), (1, [(3, 4)])], [(1, [(4, 5)])], "T(2,3) # T(3,4) # T(4,5)", "-T(2,3) # -T(3,4)", "T(2,3) # T(3,4) # -T(4,5)"),
    ([(1, [(2, 3)])], [(1, [(3, 4)]), (1, [(4, 5)])], "T(2,3) # T(3,4) # T(4,5)", "-T(2,3)", "T(2,3) # -T(3,4) # -T(4,5)"),
    # Subtraction flips every component sign in the right operand.
    ([(-1, [(2, 3), (2, 5)])], [(-1, [(3, 4), (3, 5)])], "-T(2,3; 2,5) # -T(3,4; 3,5)", "T(2,3; 2,5)", "-T(2,3; 2,5) # T(3,4; 3,5)"),
    # Repeated components and deeper cable sequences retain their full data.
    ([(1, [(2, 3)])], [(1, [(2, 3)])], "T(2,3) # T(2,3)", "-T(2,3)", "T(2,3) # -T(2,3)"),
    ([(1, [(2, 3), (2, 5), (2, 7)])], [(1, [(3, 4)])], "T(2,3; 2,5; 2,7) # T(3,4)", "-T(2,3; 2,5; 2,7)", "T(2,3; 2,5; 2,7) # -T(3,4)"),
    # Adding an explicit inverse and subtracting it exercise opposite signs.
    ([(1, [(5, 7)])], [(-1, [(5, 7)])], "T(5,7) # -T(5,7)", "-T(5,7)", "T(5,7) # T(5,7)")
])
def test_gaknot_algebraic_operations_parametric(desc1, desc2, expected_sum, expected_neg1, expected_diff):
    """Operations transform signs and component lists without simplification."""
    knot1 = GeneralizedAlgebraicKnot(desc1)
    knot2 = GeneralizedAlgebraicKnot(desc2)

    # Connected sum appends every right-hand component after every left-hand
    # component and leaves their individual signs unchanged.
    sum_knot = knot1 + knot2
    assert str(sum_knot) == expected_sum
    assert sum_knot.description == (
        _normalized_description(desc1) + _normalized_description(desc2)
    )

    # Concordance inverse flips each component sign independently but does not
    # reverse component order or cable order.
    neg_knot1 = -knot1
    assert str(neg_knot1) == expected_neg1
    assert neg_knot1.description == [
        (-sign, knot_desc)
        for sign, knot_desc in _normalized_description(desc1)
    ]

    # Subtraction is defined as connected sum with the inverse of knot2.
    diff_knot = knot1 - knot2
    assert str(diff_knot) == expected_diff
    assert diff_knot.description == (
        _normalized_description(desc1)
        + [
            (-sign, knot_desc)
            for sign, knot_desc in _normalized_description(desc2)
        ]
    )


@pytest.mark.parametrize("other", [None, "T(2,3)", [(1, [(2, 3)])]])
def test_addition_rejects_non_knots(other):
    """Connected sum requires another validated knot object."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    # Text and raw description data are not implicitly converted, since doing
    # so would bypass the explicit constructor boundary.
    with pytest.raises(TypeError, match="Can only add"):
        knot + other


# ---------------------------------------------------------------------------
# Alexander polynomial
# ---------------------------------------------------------------------------

# Fix one common polynomial ring so every expected value is compared inside
# ``ZZ[t]`` rather than through ad hoc symbolic coercions.
R_alex = PolynomialRing(ZZ, 't')
t_alex = R_alex.gen()

# The expected values exercise the three rules used by the implementation:
#
# 1. ``Delta_{K#J} = Delta_K * Delta_J``;
# 2. the stored concordance sign does not change the chosen normalization;
# 3. ``Delta_{K_(p,q)}(t) = Delta_{T(p,q)}(t) * Delta_K(t^p)``.
@pytest.mark.parametrize("desc, expected_poly", [
    # Independently familiar torus-knot polynomials provide base cases.
    ([(1, [(2, 3)])], t_alex**2 - t_alex + 1),
    ([(1, [(2, 5)])], t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1),
    ([(1, [(3, 4)])], t_alex**6 - t_alex**5 + t_alex**3 - t_alex + 1),
    # Concordance inversion/mirroring leaves this normalization unchanged.
    ([(-1, [(2, 3)])], t_alex**2 - t_alex + 1),
    # Both equal-sign and opposite-sign connected sums multiply polynomials;
    # the representation performs no cancellation at the polynomial level.
    ([(1, [(2, 3)]), (1, [(2, 3)])], (t_alex**2 - t_alex + 1)**2),
    ([(1, [(2, 3)]), (-1, [(2, 3)])], (t_alex**2 - t_alex + 1)**2),
    # Cable T(2,3;2,5): substitute t^2 into the inner trefoil polynomial and
    # multiply by the outer pattern polynomial Delta_{T(2,5)}.
    ([(1, [(2, 3), (2, 5)])], (t_alex**4 - t_alex**2 + 1) * (t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1)),
    # Reversing the cable sequence changes which polynomial receives the
    # t^2 substitution, so this is not the preceding expression reordered.
    ([(1, [(2, 5), (2, 3)])], (t_alex**8 - t_alex**6 + t_alex**4 - t_alex**2 + 1) * (t_alex**2 - t_alex + 1)),
    # A connected sum of two ordinary knots is the direct product rule.
    ([(1, [(2, 3)]), (1, [(2, 5)])], (t_alex**2 - t_alex + 1) * (t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1)),
    # Combine an iterated summand with an ordinary summand to verify that the
    # cable computation is completed before connected-sum multiplication.
    ([(1, [(2, 3), (2, 5)]), (1, [(3, 2)])], (t_alex**4 - t_alex**2 + 1) * (t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1) * (t_alex**2 - t_alex + 1))
])
def test_gaknot_alexander_polynomial_parametric(desc, expected_poly):
    """Alexander polynomials obey cabling and connected-sum formulas exactly."""
    knot = GeneralizedAlgebraicKnot(desc)
    assert knot.alexander_polynomial() == expected_poly


# ---------------------------------------------------------------------------
# Connected-sum container behavior
# ---------------------------------------------------------------------------

# ``len`` counts connected-sum components rather than cable layers.  Integer
# indexing returns a new one-component knot, while slicing returns a new knot
# containing the selected components.  Lambdas keep the fixture table compact
# while allowing each row to state a different public-container property.
@pytest.mark.parametrize("desc, test_fn", [
    # Component count is independent of the depth of any individual summand.
    ([(1, [(2, 3)])], lambda k: len(k) == 1),
    ([(1, [(2, 3)]), (1, [(3, 4)])], lambda k: len(k) == 2),
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: len(k) == 3),
    # Positive, negative, and standard negative-Python indices retain signs.
    ([(1, [(2, 3)]), (-1, [(3, 4)])], lambda k: str(k[0]) == "T(2,3)"),
    ([(1, [(2, 3)]), (-1, [(3, 4)])], lambda k: str(k[1]) == "-T(3,4)"),
    ([(1, [(2, 3)]), (-1, [(3, 4)])], lambda k: str(k[-1]) == "-T(3,4)"),
    # Slices preserve order and return full knot objects rather than raw lists.
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: str(k[0:2]) == "T(2,3) # T(3,4)"),
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: len(k[1:]) == 2),
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: str(k[:1]) == "T(2,3)"),
    # Out-of-range access translates the underlying list error to knot context.
    ([(1, [(2, 3)])], lambda k: pytest.raises(IndexError, k.__getitem__, 10))
])
def test_gaknot_container_behavior_parametric(desc, test_fn):
    """The object behaves as a container of connected-sum components."""
    knot = GeneralizedAlgebraicKnot(desc)
    result = test_fn(knot)
    # Boolean lambdas are asserted directly.  The exception lambda returns the
    # truthy context object produced by ``pytest.raises`` after checking the call.
    if result is not None:
        assert result


def test_empty_slice_is_rejected():
    """A slice cannot produce the unrepresentable empty connected sum."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    # Python list slicing itself succeeds, but constructing a knot from the
    # resulting empty description must reapply the nonempty invariant.
    with pytest.raises(ValueError, match="at least one summand"):
        knot[0:0]


# ---------------------------------------------------------------------------
# Predicates describing the knot's structural shape
# ---------------------------------------------------------------------------

# Ordinary torus knots form the one-layer subset of iterated torus knots.
# Positive and negative predicates are sign-sensitive, and all four require
# exactly one connected-sum component.  The table makes those overlaps and
# exclusions explicit rather than treating the predicates as four unrelated
# booleans.
@pytest.mark.parametrize("desc, expected_results", [
    # A positive ordinary torus knot satisfies both positive predicates.
    ([(1, [(2, 3)])], {'pos': True, 'neg': False, 'it': True, 'neg_it': False}),
    # Its concordance inverse satisfies both negative predicates.
    ([(-1, [(2, 3)])], {'pos': False, 'neg': True, 'it': False, 'neg_it': True}),
    # A genuine multi-layer positive knot is iterated but not ordinary.
    ([(1, [(2, 3), (2, 5)])], {'pos': False, 'neg': False, 'it': True, 'neg_it': False}),
    # A connected sum is not classified as one torus/iterated knot.
    ([(1, [(2, 3)]), (1, [(3, 4)])], {'pos': False, 'neg': False, 'it': False, 'neg_it': False}),
    # Repeat ordinary cases with other parameters to avoid fixture-specific logic.
    ([(1, [(3, 5)])], {'pos': True, 'neg': False, 'it': True, 'neg_it': False}),
    ([(-1, [(3, 5)])], {'pos': False, 'neg': True, 'it': False, 'neg_it': True}),
    # Cabling depth has no upper-bound special case in the iterated predicate.
    ([(1, [(2, 3), (6, 5), (30, 7)])], {'pos': False, 'neg': False, 'it': True, 'neg_it': False}),
    # Mixed signs do not change the exclusion of connected sums.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], {'pos': False, 'neg': False, 'it': False, 'neg_it': False}),
    # Another positive ordinary example completes the basic parameter coverage.
    ([(1, [(2, 7)])], {'pos': True, 'neg': False, 'it': True, 'neg_it': False}),
    # A multi-layer inverse satisfies only the broad negative-iterated predicate.
    ([(-1, [(2, 3), (2, 5)])], {'pos': False, 'neg': False, 'it': False, 'neg_it': True})
])
def test_gaknot_type_verification_parametric(desc, expected_results):
    """Shape predicates distinguish sign, layer depth, and component count."""
    knot = GeneralizedAlgebraicKnot(desc)
    assert knot.is_positive_torus_knot() == expected_results['pos']
    assert knot.is_negative_torus_knot() == expected_results['neg']
    assert knot.is_iterated_torus_knot() == expected_results['it']
    assert knot.is_neg_iterated_torus_knot() == expected_results['neg_it']


# ---------------------------------------------------------------------------
# Convenience constructors and the cabling operation
# ---------------------------------------------------------------------------

# ``torus_knot`` is shorthand for a one-component, one-pair description.  Both
# signs and several parameter choices must pass through the regular constructor.
@pytest.mark.parametrize("p, q, sign, expected_str", [
    (2, 3, 1, "T(2,3)"),
    (2, 5, -1, "-T(2,5)"),
    (3, 4, 1, "T(3,4)"),
    (3, 5, -1, "-T(3,5)"),
])
def test_torus_knot_classmethod(p, q, sign, expected_str):
    """The torus-knot constructor preserves parameters and component sign."""
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q, sign)
    assert str(knot) == expected_str


# ``iterated_torus_knot`` preserves the supplied inner-to-outer sequence; it
# must not sort or reverse cable pairs when adding the outer component wrapper.
@pytest.mark.parametrize("sequence, sign, expected_str", [
    ([(2, 3), (2, 5)], 1, "T(2,3; 2,5)"),
    ([(2, 5), (2, 3)], -1, "-T(2,5; 2,3)"),
    ([(2, 3), (2, 5), (2, 7)], 1, "T(2,3; 2,5; 2,7)"),
    ([(2, 7), (2, 5), (2, 3)], -1, "-T(2,7; 2,5; 2,3)"),
])
def test_iterated_torus_knot_classmethod(sequence, sign, expected_str):
    """The iterated constructor retains cable order and sign exactly."""
    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(sequence, sign)
    assert str(knot) == expected_str


# Cabling appends one new outermost pair.  Rows include positive and negative
# base knots and cable sequences of several initial depths to verify that no
# existing pair or sign is disturbed.
@pytest.mark.parametrize("base_knot, p, q, expected_str", [
    (GeneralizedAlgebraicKnot.torus_knot(2, 3), 2, 5, "T(2,3; 2,5)"),
    (GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5)]), 2, 7, "T(2,3; 2,5; 2,7)"),
    (GeneralizedAlgebraicKnot.torus_knot(2, 5, sign=-1), 2, 3, "-T(2,5; 2,3)"),
    (GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5), (2, 7)], sign=-1), 3, 2, "-T(2,3; 2,5; 2,7; 3,2)"),
])
def test_cable_operation(base_knot, p, q, expected_str):
    """Cabling appends an outer pattern and preserves the original sign."""
    cabled = base_knot.cable(p, q)
    # Check both readable formatting and the exact structural transformation.
    assert str(cabled) == expected_str
    assert cabled.description == [
        (
            base_knot.description[0][0],
            base_knot.description[0][1] + [(p, q)],
        )
    ]


# New cable parameters re-enter through ordinary description validation.  This
# table samples each independent rule rather than retesting every constructor
# validation row above.
@pytest.mark.parametrize("p, q, error_type, match", [
    # A noncoprime pair is a torus link, p=1 violates positivity, and a float
    # violates the integer parameter contract.
    (2, 4, ValueError, "relatively prime"),
    (1, 3, ValueError, "must be > 1"),
    (2.5, 3, TypeError, "must be integers"),
])
def test_cable_validates_new_parameters(p, q, error_type, match):
    """The cabling convenience method cannot bypass pair validation."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    with pytest.raises(error_type, match=match):
        knot.cable(p, q)


def test_cable_restriction():
    """Cabling is defined for one iterated summand, not a connected sum."""
    # With two summands there is no unambiguous component to receive the new
    # outer pattern, so the operation must fail rather than guess.
    sum_knot = GeneralizedAlgebraicKnot.torus_knot(2, 3) + GeneralizedAlgebraicKnot.torus_knot(2, 5)
    with pytest.raises(ValueError, match="Cabling is only supported for iterated torus knots"):
        sum_knot.cable(2, 7)
    
    # The restriction concerns component count, not existing cable depth.  A
    # one-summand iterated knot can always receive another valid outer layer.
    it_knot = GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5)])
    assert str(it_knot.cable(2, 7)) == "T(2,3; 2,5; 2,7)"
