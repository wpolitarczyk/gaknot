"""Tests for the Levine--Tristram signature constructors.

The tests deliberately use several kinds of oracle:

* known classical signature values at ``theta = 1/2``;
* structural properties such as vanishing total jump and ``T(p,q)=T(q,p)``;
* exact cabling formulas assembled outside the optimized counter accumulator;
* exact additivity for signed connected sums; and
* explicit validation contracts, including exception type and index context.

All mathematically distinguished points are represented by Sage rationals.
This avoids allowing floating-point rounding to move an evaluation from a jump
point to one of its neighboring constant intervals.
"""

import pytest
from sage.all import QQ

from gaknot.invariants.LT_signature import (
    LT_signature_torus_knot,
    LT_signature_iterated_torus_knot,
    LT_signature_generalized_algebraic_knot,
    reparametrize,
)


def _iterated_description_id(desc):
    """Build a readable pytest ID that displays the successive cable layers."""
    # The first pair names the base torus knot.  Every subsequent pair is shown
    # as an outer cabling operation in the same order as the input description.
    pairs = [f"({p},{q})" for p, q in desc]
    return f"T{pairs[0]}" + "".join(
        f"-cable{pair}" for pair in pairs[1:]
    )


def _expected_iterated_signature(desc):
    r"""Assemble the cabling formula independently of the optimized routine.

    For each pair at index ``i``, the winding factor is the product of the
    ``p`` parameters in all outer layers.  Iterating from outside to inside
    makes this product easy to maintain.  Equality of ``SignatureFunction``
    objects compares their complete jump counters, including discontinuities.
    """
    expected = None
    # The outermost torus pattern appears without reparametrization.
    winding = 1

    for p, q in reversed(desc):
        # This uses the public elementary operations rather than the optimized
        # iterated counter accumulator, so it exercises a distinct code path.
        component = reparametrize(LT_signature_torus_knot(p, q), winding)
        # Avoid Python's numeric sum() identity: SignatureFunction is not meant
        # to be added to the integer zero.
        expected = component if expected is None else expected + component
        # Every component farther inside is reparametrized by this cable's p.
        winding *= p

    return expected


def _expected_generalized_signature(desc):
    """Add signed public iterated signatures to form an exact reference value."""
    expected = None

    for sign, knot_desc in desc:
        # Scalar multiplication negates all jump weights when sign == -1.
        component = sign * LT_signature_iterated_torus_knot(knot_desc)
        # As above, initialize from the first SignatureFunction rather than 0.
        expected = component if expected is None else expected + component

    return expected


# ---------------------------------------------------------------------------
# Elementary torus knots
# ---------------------------------------------------------------------------

# These are independently known values of the ordinary knot signature, which
# is the Levine--Tristram signature at theta = 1/2.  The trefoil T(2,3) is
# omitted here because tests/test_signature.py checks its complete step function.
@pytest.mark.parametrize("p, q, expected_sig", [
    # Two-strand torus knots provide a simple increasing-genus family.
    (2, 5, -4),
    (2, 7, -6),
    # Higher-strand examples prevent the table from encoding only the T(2,q)
    # closed form and include cases whose signature is not minus twice genus.
    (3, 4, -6),
    (3, 5, -8),
    (3, 7, -8),
    (4, 5, -8),
    (4, 7, -14),
    (5, 6, -16),
    (7, 8, -30)
])
def test_lt_signature_torus_knot_basic(p, q, expected_sig):
    sig = LT_signature_torus_knot(p, q)

    # There are no jumps before theta = 0, so the normalized signature starts
    # at zero.  A nonzero value here would indicate a bad endpoint convention.
    assert sig(0) == 0

    # Periodicity on the unit circle requires all signed jump weights to cancel
    # after one complete turn.  This is useful but not sufficient by itself.
    assert sig.total_sign_jump() == 0

    # Use an exact rational and compare directly.  Coercing with int() could
    # conceal an erroneous nonintegral result by truncating it.
    assert sig(QQ(1) / 2) == expected_sig


# The error table is grouped by the validation rule that should reject it.
@pytest.mark.parametrize("p, q, error_type, match", [
    # Noncoprime pairs describe torus links and make inverse_mod(p, q) invalid.
    (3, 6, ValueError, "relatively prime"),
    (2, 4, ValueError, "relatively prime"),
    (10, 15, ValueError, "relatively prime"),
    # Zero, one, and negative values lie outside the supported positive-knot API.
    (1, 3, ValueError, "must be >1"),
    (3, 1, ValueError, "must be >1"),
    (0, 5, ValueError, "must be >1"),
    (-2, 3, ValueError, "must be >1"),
    # Nonintegers must fail before modular arithmetic begins.
    (2.5, 3, TypeError, "have to be integers"),
    (2, "3", TypeError, "have to be integers"),
    (None, 3, TypeError, "have to be integers"),
    # bool is an int subclass in Python, so it needs explicit regression cases.
    (True, 3, TypeError, "have to be integers"),
    (2, False, TypeError, "have to be integers"),
])
def test_lt_signature_torus_knot_errors_parametric(p, q, error_type, match):
    # Matching part of the message verifies that the public validator rejected
    # the input, rather than an unrelated low-level Sage operation.
    with pytest.raises(error_type, match=match):
        LT_signature_torus_knot(p, q)


# T(p,q) and T(q,p) are isotopic.  Each unordered pair appears once because the
# assertion itself already evaluates both orders.
@pytest.mark.parametrize("p, q", [
    (2, 3),
    (2, 5),
    (3, 4),
    (3, 7),
    (4, 5),
    (5, 6),
    (7, 8),
    (2, 11),
    (3, 10),
    (5, 12),
])
def test_lt_signature_torus_knot_symmetry_parametric(p, q):
    sig1 = LT_signature_torus_knot(p, q)
    sig2 = LT_signature_torus_knot(q, p)

    # SignatureFunction equality compares every exact jump and its weight.
    assert sig1 == sig2


# ---------------------------------------------------------------------------
# Iterated torus knots
# ---------------------------------------------------------------------------

# The table mixes two- and three-layer descriptions, algebraic-style cables,
# small winding parameters, and reversed-looking base pairs.
@pytest.mark.parametrize("desc", [
    # Two-layer examples with distinct winding products and torus parameters.
    ([(2, 3), (6, 5)]),
    ([(2, 5), (10, 3)]),
    ([(3, 4), (12, 5)]),
    # A small winding-number example formerly covered only at five sample points.
    ([(2, 3), (2, 5)]),
    # This case checks that p and q are used in their documented roles even
    # when the base pair is written in the symmetric order T(3,2).
    ([(3, 2), (2, 3)]),
    # Three layers exercise accumulation of more than one outer p-factor.
    ([(2, 3), (6, 5), (30, 7)]),
    # Additional two-layer cases broaden the set of jump coincidences and
    # cancellation patterns seen by the counter accumulator.
    ([(2, 7), (14, 3)]),
    ([(3, 5), (15, 2)]),
    ([(2, 3), (2, 7)]),
    ([(2, 5), (2, 3)])
], ids=_iterated_description_id)
def test_lt_signature_iterated_torus_knot_parametric(desc):
    iterated_sig = LT_signature_iterated_torus_knot(desc)
    expected_sig = _expected_iterated_signature(desc)

    # This is the primary mathematical assertion: it compares the complete
    # sparse function, not merely a few regular evaluation points.
    assert iterated_sig == expected_sig
    # Retain periodic jump balance as a secondary structural invariant.
    assert iterated_sig.total_sign_jump() == 0


# These cases walk from malformed outer structure toward progressively deeper
# numerical errors, ensuring the validator reports the correct cable index.
@pytest.mark.parametrize("desc, error_type, match", [
    # Invalid outer container and missing base knot.
    ("not a sequence", TypeError, "list or tuple"),
    ([], ValueError, "at least one cabling pair"),
    # Each layer must unpack to exactly two values.
    ([(2,)], ValueError, r"pair \(p, q\)"),
    ([(2, 3, 5)], ValueError, r"pair \(p, q\)"),
    # Parameter types, including the special bool-as-int case.
    ([(2.5, 3)], TypeError, "have to be integers"),
    ([(2, "3")], TypeError, "have to be integers"),
    ([(True, 3)], TypeError, "have to be integers"),
    # Integer values that fail positivity or coprimality.
    ([(1, 3)], ValueError, "must be >1"),
    ([(2, 0)], ValueError, "must be >1"),
    ([(2, 4)], ValueError, "relatively prime"),
    # A valid first pair must not hide an invalid later cabling layer.
    ([(2, 3), (6, 9)], ValueError, "index 1.*relatively prime"),
])
def test_lt_signature_iterated_torus_knot_errors(desc, error_type, match):
    # Error category and message context form part of the public contract.
    with pytest.raises(error_type, match=match):
        LT_signature_iterated_torus_knot(desc)


def test_lt_signature_iterated_torus_knot_accepts_tuple_description():
    # Tuples are useful to callers who keep knot descriptions immutable.
    tuple_description = ((2, 3), (2, 5))

    # Container choice must not affect any jump location or weight.
    assert LT_signature_iterated_torus_knot(tuple_description) == (
        LT_signature_iterated_torus_knot(list(tuple_description))
    )


# ---------------------------------------------------------------------------
# Generalized algebraic knots (signed connected sums)
# ---------------------------------------------------------------------------

# Generalized descriptions add an outer structural and sign-validation layer
# around the iterated descriptions tested above.
@pytest.mark.parametrize("desc, error_type, match", [
    # Invalid outer container, empty sum, and malformed summand shape.
    ("not a sequence", TypeError, "list or tuple"),
    ([], ValueError, "at least one summand"),
    ([(1,)], ValueError, "must be a pair"),
    # Only the integers +1 and -1 encode the two allowed orientations.
    ([(0, [(2, 3)])], ValueError, "Sign at index 0"),
    ([(True, [(2, 3)])], ValueError, "Sign at index 0"),
    # Nested failures must preserve both error type and summand context.
    ([(1, [])], ValueError, "index 0.*at least one cabling pair"),
    ([(1, "not a sequence")], TypeError, "index 0.*list or tuple"),
    ([(1, [(2, 4)])], ValueError, "index 0.*relatively prime"),
    ([(1, [(2, 3), (6, 9)])], ValueError, "index 0.*index 1"),
])
def test_lt_signature_generalized_algebraic_knot_errors(
    desc,
    error_type,
    match,
):
    # The message match ensures nested failures identify the outer summand and,
    # where applicable, the inner cabling layer.
    with pytest.raises(error_type, match=match):
        LT_signature_generalized_algebraic_knot(desc)


def test_lt_signature_generalized_algebraic_knot_accepts_tuple_description():
    # Exercise tuples at both the connected-sum and iterated-description levels.
    tuple_description = (
        (1, ((2, 3),)),
        (-1, ((2, 5),)),
    )

    # The tuple and list encodings represent precisely the same signed sum.
    assert LT_signature_generalized_algebraic_knot(tuple_description) == (
        LT_signature_generalized_algebraic_knot(list(tuple_description))
    )


# The table includes exact cancellation, nonzero sums, multi-layer summands,
# and a known algebraically slice combination.  ``expected_zero`` records the
# independent high-level property in addition to exact additivity below.
@pytest.mark.parametrize("desc, expected_zero", [
    # A knot summed with its inverse cancels coefficient by coefficient.
    ([(1, [(2, 5)]), (-1, [(2, 5)])], True),
    # A nontrivial known algebraically slice combination; cancellation is spread
    # across four different iterated/torus summands rather than identical pairs.
    ([
        (1, [(2, 3), (5, 2)]),
        (1, [(3, 2)]),
        (1, [(5, 3)]),
        (-1, [(6, 5)])
    ], True),
    # A single nontrivial summand establishes the nonzero baseline.
    ([(1, [(2, 3)])], False),
    # Two positive, distinct summands must not be mistaken for cancellation.
    ([(1, [(2, 3)]), (1, [(3, 4)])], False),
    # Exact cancellation also holds for a multi-layer iterated knot.
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3), (6, 5)])], True),
    # Two independently cancelling pairs test accumulation across four entries.
    ([
        (1, [(2, 3)]), (-1, [(2, 3)]),
        (1, [(3, 4)]), (-1, [(3, 4)])
    ], True),
    # The two T(2,3) terms cancel, leaving the nonzero T(3,4) contribution.
    ([(1, [(2, 3)]), (1, [(3, 4)]), (-1, [(2, 3)])], False),
    # A second single-component example guards against trefoil-specific behavior.
    ([(1, [(2, 7)])], False),
    # Cancellation is independent of the order in which inverse terms appear.
    ([(-1, [(2, 3)]), (1, [(2, 3)])], True)
])
def test_lt_signature_generalized_algebraic_knot_parametric(desc, expected_zero):
    sig = LT_signature_generalized_algebraic_knot(desc)
    expected_sig = _expected_generalized_signature(desc)

    # Compare the entire jump counter with the signed sum assembled through the
    # public iterated API.  Any wrong nonzero function fails this assertion.
    assert sig == expected_sig
    # This secondary assertion makes the intended cancellation behavior visible
    # in the test data and gives failures a direct algebraic interpretation.
    assert sig.is_zero_everywhere() == expected_zero
