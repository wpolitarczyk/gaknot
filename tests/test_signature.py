import pytest
import warnings
import time
from sage.all import QQ
from gaknot import GeneralizedAlgebraicKnot, SignatureFunction
from gaknot.invariants.LT_signature import LT_signature_torus_knot, LT_signature_iterated_torus_knot

# Suppress DeprecationWarnings from SageMath and other libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*superseded by LazyCombinatorialSpecies.*")
warnings.filterwarnings('ignore', message=".*Importing .* from here is deprecated.*")

def test_torus_knot_signature_values():
    # T(2,3) is the trefoil knot. 
    # Jumps at 1/6 (value -1) and 5/6 (value +1)
    # Sigma(t) = 2 * sum(jumps before t) + jump at t
    sig = LT_signature_torus_knot(2, 3)
    
    # Before 1/6: 0
    assert sig(0.1) == 0
    # At 1/6: -1
    assert sig(QQ(1)/6) == -1
    # Between 1/6 and 5/6: 2 * (-1) = -2
    assert sig(0.5) == -2
    # At 5/6: 2*(-1) + (+1) = -1
    assert sig(QQ(5)/6) == -1
    # After 5/6: 2*(-1) + 2*(1) = 0
    assert sig(0.9) == 0

def test_connected_sum_signature():
    # T(2,3) # -T(2,3) should be zero everywhere
    K = GeneralizedAlgebraicKnot([(1, [(2,3)]), (-1, [(2,3)])])
    sig = K.signature()
    assert sig.is_zero_everywhere()

def test_iterated_torus_knot_reparametrization():
    # T(2,3; 2,5) signature should match formula
    # sigma(t) = sigma_T(2,3)(2t) + sigma_T(2,5)(t)
    desc = [(2,3), (2,5)]
    sig_iterated = LT_signature_iterated_torus_knot(desc)
    
    sig_23 = LT_signature_torus_knot(2,3)
    sig_25 = LT_signature_torus_knot(2,5)
    
    # Test at some points
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        expected = sig_23(2*t) + sig_25(t)
        assert sig_iterated(t) == expected, f"Failed at t={t}"

def test_large_knot_performance():
    # A knot with many jumps
    # T(91, 874) has ~80000 jumps
    start = time.time()
    sig = LT_signature_torus_knot(91, 874)
    end = time.time()
    # print(f"Time to compute T(91, 874): {end - start:.4f}s")
    assert end - start < 1.0 # Should be very fast now
    
    # Evaluation performance
    start = time.time()
    for i in range(1000):
        _ = sig(i/1000)
    end = time.time()
    # print(f"Time for 1000 evaluations: {end - start:.4f}s")
    assert end - start < 0.2 # Should be O(log N) now
