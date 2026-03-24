
import time
from gaknot_lib.gaknot import GeneralizedAlgebraicKnot
from gaknot_lib.signature import SignatureFunction
from gaknot_lib.LT_signature import LT_signature_torus_knot, LT_signature_iterated_torus_knot
import os

# Mock the load for example_generator.sage
# Since we are in a script, we can't 'load' but we can read and exec
with open("research/example_generator.sage", "r") as f:
    exec(f.read())

def test_single():
    # 4-layer random knot
    K_desc = gen_iterated(size=4, size_range=40)
    print(f"Testing K: {K_desc}")
    
    start = time.time()
    J = slicen(K_desc)
    print(f"slicen computed in {time.time() - start:.4f}s")
    
    L_desc = [(1, K_desc)] + J
    print(f"Connected sum L: {L_desc}")
    K_total = GeneralizedAlgebraicKnot(L_desc)
    
    start = time.time()
    sig = K_total.signature()
    print(f"Signature computed in {time.time() - start:.4f}s. Jumps: {len(sig.jumps_counter)}")
    
    start = time.time()
    is_zero = sig.is_zero_everywhere()
    print(f"is_zero_everywhere computed in {time.time() - start:.4f}s")
    
    assert is_zero, "Signature should be zero"
    print("SUCCESS")

if __name__ == "__main__":
    test_single()
