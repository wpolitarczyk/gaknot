import pytest
from sage.all import PolynomialRing, ZZ
from gaknot import GeneralizedAlgebraicKnot

@pytest.mark.parametrize("desc, expected_str", [
    ([(1, [(2, 3)])], "T(2,3)"),
    ([(-1, [(2, 3)])], "-T(2,3)"),
    ([(1, [(2, 3), (2, 5)])], "T(2,3; 2,5)"),
    ([(-1, [(2, 3), (2, 5)])], "-T(2,3; 2,5)"),
    ([(1, [(2, 3)]), (1, [(3, 4)])], "T(2,3) # T(3,4)"),
    ([(1, [(2, 3)]), (-1, [(3, 4)])], "T(2,3) # -T(3,4)"),
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3)])], "T(2,3; 6,5) # -T(2,3)"),
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3), (6, 7)])], "T(2,3; 6,5) # -T(2,3; 6,7)"),
    ([(1, [(2, 3)]), (1, [(3, 5)]), (1, [(5, 7)])], "T(2,3) # T(3,5) # T(5,7)"),
    ([(1, [(2, 3), (6, 5), (30, 7)])], "T(2,3; 6,5; 30,7)")
])
def test_gaknot_basic_functionality(desc, expected_str):
    knot = GeneralizedAlgebraicKnot(desc)
    assert str(knot) == expected_str
    assert knot.description == desc

@pytest.mark.parametrize("desc, error_type, match", [
    # Not a list/tuple
    ("not a list", TypeError, "must be a list or tuple"),
    # Element not a pair
    ([(1, [(2, 3)]), (1,)], ValueError, "must be a pair"),
    # Bad sign
    ([(2, [(2, 3)])], ValueError, "Sign at index 0 must be 1 or -1"),
    # Knot description not a list
    ([(1, "not a list")], TypeError, "must be a list or tuple"),
    # Cable parameter not a pair
    ([(1, [(2, 3, 4)])], ValueError, r"must be a pair \(p, q\)"),
    # Parameters not integers
    ([(1, [(2.5, 3)])], TypeError, "must be integers"),
    # p <= 1
    ([(1, [(1, 3)])], ValueError, "must be > 1"),
    # q <= 1
    ([(1, [(3, 0)])], ValueError, "must be > 1"),
    # Not relatively prime (2, 4)
    ([(1, [(2, 4)])], ValueError, "relatively prime"),
    # Not relatively prime (6, 9) in a deeper layer
    ([(1, [(2, 3), (6, 9)])], ValueError, "relatively prime")
])
def test_gaknot_validation_parametric(desc, error_type, match):
    with pytest.raises(error_type, match=match):
        GeneralizedAlgebraicKnot(desc)

@pytest.mark.parametrize("desc1, desc2, expected_sum, expected_neg1, expected_diff", [
    ([(1, [(2, 3)])], [(1, [(3, 4)])], "T(2,3) # T(3,4)", "-T(2,3)", "T(2,3) # -T(3,4)"),
    ([(1, [(2, 5)])], [(1, [(2, 3), (6, 5)])], "T(2,5) # T(2,3; 6,5)", "-T(2,5)", "T(2,5) # -T(2,3; 6,5)"),
    ([(-1, [(2, 3)])], [(1, [(3, 4)])], "-T(2,3) # T(3,4)", "T(2,3)", "-T(2,3) # -T(3,4)"),
    ([(1, [(2, 3), (6, 5)])], [(-1, [(2, 3), (6, 5)])], "T(2,3; 6,5) # -T(2,3; 6,5)", "-T(2,3; 6,5)", "T(2,3; 6,5) # T(2,3; 6,5)"),
    ([(1, [(2, 3)]), (1, [(3, 4)])], [(1, [(4, 5)])], "T(2,3) # T(3,4) # T(4,5)", "-T(2,3) # -T(3,4)", "T(2,3) # T(3,4) # -T(4,5)"),
    ([(1, [(2, 3)])], [(1, [(3, 4)]), (1, [(4, 5)])], "T(2,3) # T(3,4) # T(4,5)", "-T(2,3)", "T(2,3) # -T(3,4) # -T(4,5)"),
    ([(-1, [(2, 3), (2, 5)])], [(-1, [(3, 4), (3, 5)])], "-T(2,3; 2,5) # -T(3,4; 3,5)", "T(2,3; 2,5)", "-T(2,3; 2,5) # T(3,4; 3,5)"),
    ([(1, [(2, 3)])], [(1, [(2, 3)])], "T(2,3) # T(2,3)", "-T(2,3)", "T(2,3) # -T(2,3)"),
    ([(1, [(2, 3), (2, 5), (2, 7)])], [(1, [(3, 4)])], "T(2,3; 2,5; 2,7) # T(3,4)", "-T(2,3; 2,5; 2,7)", "T(2,3; 2,5; 2,7) # -T(3,4)"),
    ([(1, [(5, 7)])], [(-1, [(5, 7)])], "T(5,7) # -T(5,7)", "-T(5,7)", "T(5,7) # T(5,7)")
])
def test_gaknot_algebraic_operations_parametric(desc1, desc2, expected_sum, expected_neg1, expected_diff):
    knot1 = GeneralizedAlgebraicKnot(desc1)
    knot2 = GeneralizedAlgebraicKnot(desc2)

    sum_knot = knot1 + knot2
    assert str(sum_knot) == expected_sum

    neg_knot1 = -knot1
    assert str(neg_knot1) == expected_neg1

    diff_knot = knot1 - knot2
    assert str(diff_knot) == expected_diff

# Setup Polynomial Ring for Alexander Polynomial tests
R_alex = PolynomialRing(ZZ, 't')
t_alex = R_alex.gen()

@pytest.mark.parametrize("desc, expected_poly", [
    ([(1, [(2, 3)])], t_alex**2 - t_alex + 1),
    ([(1, [(2, 5)])], t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1),
    ([(1, [(3, 4)])], t_alex**6 - t_alex**5 + t_alex**3 - t_alex + 1),
    ([(-1, [(2, 3)])], t_alex**2 - t_alex + 1),
    ([(1, [(2, 3)]), (1, [(2, 3)])], (t_alex**2 - t_alex + 1)**2),
    ([(1, [(2, 3)]), (-1, [(2, 3)])], (t_alex**2 - t_alex + 1)**2),
    ([(1, [(2, 3), (2, 5)])], (t_alex**4 - t_alex**2 + 1) * (t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1)),
    ([(1, [(2, 5), (2, 3)])], (t_alex**8 - t_alex**6 + t_alex**4 - t_alex**2 + 1) * (t_alex**2 - t_alex + 1)),
    ([(1, [(2, 3)]), (1, [(2, 5)])], (t_alex**2 - t_alex + 1) * (t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1)),
    ([(1, [(2, 3), (2, 5)]), (1, [(3, 2)])], (t_alex**4 - t_alex**2 + 1) * (t_alex**4 - t_alex**3 + t_alex**2 - t_alex + 1) * (t_alex**2 - t_alex + 1))
])
def test_gaknot_alexander_polynomial_parametric(desc, expected_poly):
    knot = GeneralizedAlgebraicKnot(desc)
    assert knot.alexander_polynomial() == expected_poly

@pytest.mark.parametrize("desc, test_fn", [
    # len tests
    ([(1, [(2, 3)])], lambda k: len(k) == 1),
    ([(1, [(2, 3)]), (1, [(3, 4)])], lambda k: len(k) == 2),
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: len(k) == 3),
    # indexing tests
    ([(1, [(2, 3)]), (-1, [(3, 4)])], lambda k: str(k[0]) == "T(2,3)"),
    ([(1, [(2, 3)]), (-1, [(3, 4)])], lambda k: str(k[1]) == "-T(3,4)"),
    ([(1, [(2, 3)]), (-1, [(3, 4)])], lambda k: str(k[-1]) == "-T(3,4)"),
    # slicing tests
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: str(k[0:2]) == "T(2,3) # T(3,4)"),
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: len(k[1:]) == 2),
    ([(1, [(2, 3)]), (1, [(3, 4)]), (1, [(4, 5)])], lambda k: str(k[:1]) == "T(2,3)"),
    # exception test
    ([(1, [(2, 3)])], lambda k: pytest.raises(IndexError, k.__getitem__, 10))
])
def test_gaknot_container_behavior_parametric(desc, test_fn):
    knot = GeneralizedAlgebraicKnot(desc)
    result = test_fn(knot)
    if result is not None:
        assert result

@pytest.mark.parametrize("desc, expected_results", [
    # Positive Torus Knot
    ([(1, [(2, 3)])], {'pos': True, 'neg': False, 'it': True}),
    # Negative Torus Knot
    ([(-1, [(2, 3)])], {'pos': False, 'neg': True, 'it': False}),
    # Positive Iterated Torus Knot (not basic)
    ([(1, [(2, 3), (2, 5)])], {'pos': False, 'neg': False, 'it': True}),
    # Connected sum (not a single torus/iterated knot)
    ([(1, [(2, 3)]), (1, [(3, 4)])], {'pos': False, 'neg': False, 'it': False}),
    # Single positive torus knot (different parameters)
    ([(1, [(3, 5)])], {'pos': True, 'neg': False, 'it': True}),
    # Single negative torus knot (different parameters)
    ([(-1, [(3, 5)])], {'pos': False, 'neg': True, 'it': False}),
    # Iterated torus knot with deeper layering
    ([(1, [(2, 3), (6, 5), (30, 7)])], {'pos': False, 'neg': False, 'it': True}),
    # Connected sum with negative component
    ([(1, [(2, 3)]), (-1, [(2, 3)])], {'pos': False, 'neg': False, 'it': False}),
    # Single positive torus knot (another one)
    ([(1, [(2, 7)])], {'pos': True, 'neg': False, 'it': True}),
    # Single negative iterated torus knot
    ([(-1, [(2, 3), (2, 5)])], {'pos': False, 'neg': False, 'it': False})
])
def test_gaknot_type_verification_parametric(desc, expected_results):
    knot = GeneralizedAlgebraicKnot(desc)
    assert knot.is_positive_torus_knot() == expected_results['pos']
    assert knot.is_negative_torus_knot() == expected_results['neg']
    assert knot.is_iterated_torus_knot() == expected_results['it']
