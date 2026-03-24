import time
from gaknot_lib.gaknot import GeneralizedAlgebraicKnot
from gaknot_lib.signature import SignatureFunction
from gaknot_lib.LT_signature import LT_signature_torus_knot, LT_signature_iterated_torus_knot

def benchmark():
    # Test a simple torus knot
    start = time.time()
    sig1 = LT_signature_torus_knot(91, 874)
    print(f"LT_signature_torus_knot(91, 874) took {time.time() - start:.4f}s. Jumps: {len(sig1.jumps_counter)}")

    # Test an iterated torus knot with 2 layers
    start = time.time()
    sig2 = LT_signature_iterated_torus_knot([(91, 874), (60, 529)])
    print(f"LT_signature_iterated_torus_knot([(91, 874), (60, 529)]) took {time.time() - start:.4f}s. Jumps: {len(sig2.jumps_counter)}")

    # Test an iterated torus knot with 3 layers
    start = time.time()
    try:
        sig3 = LT_signature_iterated_torus_knot([(91, 874), (60, 529), (66, 329)])
        print(f"LT_signature_iterated_torus_knot([(91, 874), (60, 529), (66, 329)]) took {time.time() - start:.4f}s. Jumps: {len(sig3.jumps_counter)}")
    except Exception as e:
        print(f"3-layer failed or too slow: {e}")

if __name__ == "__main__":
    benchmark()
