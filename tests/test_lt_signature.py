import pytest
import warnings
from gaknot.invariants.LT_signature import (
    LT_signature_torus_knot,
    LT_signature_iterated_torus_knot,
    LT_signature_generalized_algebraic_knot
)

# Suppress DeprecationWarnings from SageMath and other libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*superseded by LazyCombinatorialSpecies.*")
warnings.filterwarnings('ignore', message=".*Importing .* from here is deprecated.*")

@pytest.mark.parametrize("p, q, expected_sig", [
    (2, 3, -2),
    (2, 5, -4),
    (2, 7, -6),
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
    assert sig(0) == 0
    assert sig.total_sign_jump() == 0

    # Check the signature value at 1/2
    assert int(sig(0.5)) == expected_sig

@pytest.mark.parametrize("p, q, error_type, match", [
    (3, 6, ValueError, "relatively prime"),
    (2, 4, ValueError, "relatively prime"),
    (10, 15, ValueError, "relatively prime"),
    (1, 3, ValueError, "must be >1"),
    (3, 1, ValueError, "must be >1"),
    (0, 5, ValueError, "must be >1"),
    (-2, 3, ValueError, "must be >1"),
    (2.5, 3, TypeError, "have to be integers"),
    (2, "3", TypeError, "have to be integers"),
    (None, 3, TypeError, "have to be integers")
])
def test_lt_signature_torus_knot_errors_parametric(p, q, error_type, match):
    with pytest.raises(error_type, match=match):
        LT_signature_torus_knot(p, q)

@pytest.mark.parametrize("p, q", [
    (2, 3), (3, 2),
    (2, 5), (5, 2),
    (3, 4), (4, 3),
    (3, 7), (7, 3),
    (4, 5), (5, 4),
    (5, 6), (6, 5),
    (7, 8), (8, 7),
    (2, 11), (11, 2),
    (3, 10), (10, 3),
    (5, 12), (12, 5)
])
def test_lt_signature_torus_knot_symmetry_parametric(p, q):
    sig1 = LT_signature_torus_knot(p, q)
    sig2 = LT_signature_torus_knot(q, p)
    assert sig1 == sig2

@pytest.mark.parametrize("desc", [
    ([(2, 3), (6, 5)]),
    ([(2, 5), (10, 3)]),
    ([(3, 4), (12, 5)]),
    ([(2, 3), (2, 5)]),
    ([(3, 2), (2, 3)]),
    ([(2, 3), (6, 5), (30, 7)]),
    ([(2, 7), (14, 3)]),
    ([(3, 5), (15, 2)]),
    ([(2, 3), (2, 7)]),
    ([(2, 5), (2, 3)])
])
def test_lt_signature_iterated_torus_knot_parametric(desc):
    iterated_sig = LT_signature_iterated_torus_knot(desc)
    assert iterated_sig.total_sign_jump() == 0

@pytest.mark.parametrize("desc, expected_zero", [
    # T(2,3) # -T(2,3)
    ([(1, [(2, 3)]), (-1, [(2, 3)])], True),
    # T(2,5) # -T(2,5)
    ([(1, [(2, 5)]), (-1, [(2, 5)])], True),
    # Complex known algebraically slice
    ([
        (1, [(2, 3), (5, 2)]),
        (1, [(3, 2)]),
        (1, [(5, 3)]),
        (-1, [(6, 5)])
    ], True),
    # Single T(2,3) (not zero)
    ([(1, [(2, 3)])], False),
    # T(2,3) # T(3,4) (not zero)
    ([(1, [(2, 3)]), (1, [(3, 4)])], False),
    # Concordance inverse sum
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3), (6, 5)])], True),
    # Sum of two slice candidates (should be slice if both are)
    ([
        (1, [(2, 3)]), (-1, [(2, 3)]),
        (1, [(3, 4)]), (-1, [(3, 4)])
    ], True),
    # Three summands (not zero)
    ([(1, [(2, 3)]), (1, [(3, 4)]), (-1, [(2, 3)])], False),
    # Empty (technically valid but let's test a simple single component)
    ([(1, [(2, 7)])], False),
    # -T(2,3) # T(2,3)
    ([(-1, [(2, 3)]), (1, [(2, 3)])], True)
])
def test_lt_signature_generalized_algebraic_knot_parametric(desc, expected_zero):
    sig = LT_signature_generalized_algebraic_knot(desc)
    if expected_zero:
        assert sig.is_zero_everywhere()
    else:
        assert not sig.is_zero_everywhere()
