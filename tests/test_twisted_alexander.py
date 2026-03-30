import pytest
from sage.all import QQ, CyclotomicField, PolynomialRing
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.character import Character
from gaknot.invariants.twisted_alexander import twisted_alexander_torus_knot

@pytest.mark.parametrize("p, q, expected_str", [
    (2, 3, "(-t^2 - t - 1)/(t - 1)"),
    (2, 5, "(-t^4 - t^3 - t^2 - t - 1)/(t - 1)"),
    (3, 2, "(t^2 + 2*t + 1)/(t - 1)"),
    (3, 4, "(t^6 + 2*t^5 + 3*t^4 + 4*t^3 + 3*t^2 + 2*t + 1)/(t - 1)"),
    (5, 2, "(t^4 + 4*t^3 + 6*t^2 + 4*t + 1)/(t - 1)"),
    (2, 7, "(-t^6 - t^5 - t^4 - t^3 - t^2 - t - 1)/(t - 1)"),
    (4, 3, "(-t^6 - 3*t^5 - 6*t^4 - 7*t^3 - 6*t^2 - 3*t - 1)/(t - 1)"),
    (5, 4, "(t^12 + 4*t^11 + 10*t^10 + 20*t^9 + 31*t^8 + 40*t^7 + 44*t^6 + 40*t^5 + 31*t^4 + 20*t^3 + 10*t^2 + 4*t + 1)/(t - 1)"),
    (3, 5, "(t^8 + 2*t^7 + 3*t^6 + 4*t^5 + 5*t^4 + 4*t^3 + 3*t^2 + 2*t + 1)/(t - 1)"),
    (2, 9, "(-t^8 - t^7 - t^6 - t^5 - t^4 - t^3 - t^2 - t - 1)/(t - 1)")
])
def test_twisted_alexander_torus_knot_trivial(p, q, expected_str):
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q)
    h1 = BranchedCoverHomology(knot, p)
    # Trivial character: all zeros
    char_values = [[[0] * len(h1.decomposition[0]['layers'][0]['base_factors'])]]
    char = Character(h1, char_values)
    
    poly = twisted_alexander_torus_knot(knot, char)
    
    K = CyclotomicField(q)
    R = PolynomialRing(K, 't')
    t = R.gen()
    # Sage evaluates the string to a rational function in the correct ring
    expected = R.fraction_field()(expected_str)
    assert poly == expected

@pytest.mark.parametrize("p, q, x_values, expected_str", [
    (2, 3, [1/3], "-t + 1"),
    (2, 5, [1/5], "-t^3 + (zeta5^3 + zeta5^2 + 1)*t^2 + (-zeta5^3 - zeta5^2 - 1)*t + 1"),
    (3, 2, [1/2, 1/2], "t - 1"),
    (2, 3, [2/3], "-t + 1"),
    (3, 4, [1/4, 0], "t^5 + t^4 - t - 1"),
    (2, 7, [1/7], "-t^5 + (zeta7^5 + zeta7^4 + zeta7^3 + zeta7^2 + 1)*t^4 + (-zeta7^5 - zeta7^2 - 1)*t^3 + (zeta7^5 + zeta7^2 + 1)*t^2 + (-zeta7^5 - zeta7^4 - zeta7^3 - zeta7^2 - 1)*t + 1"),
    (4, 3, [1/3, 0, 0], "-t^5 - t^4 - t^3 + t^2 + t + 1"),
    (5, 2, [1/2, 1/2, 1/2, 1/2], "t^3 - 3*t^2 + 3*t - 1"),
    (2, 9, [1/9], "-t^7 + (zeta9^5 + zeta9^2 - zeta9)*t^6 + (zeta9^4 - zeta9^2 + zeta9 - 1)*t^5 + (zeta9^5 + zeta9^2 - zeta9 + 1)*t^4 + (-zeta9^5 - zeta9^2 + zeta9 - 1)*t^3 + (-zeta9^4 + zeta9^2 - zeta9 + 1)*t^2 + (-zeta9^5 - zeta9^2 + zeta9)*t + 1"),
    (3, 5, [1/5, 2/5], "t^7 + (zeta5^3 - zeta5^2 - zeta5 - 1)*t^6 + (zeta5^3 + 2*zeta5^2 + 3*zeta5)*t^5 + (-4*zeta5^3 - 3*zeta5^2 - 2*zeta5 - 1)*t^4 + (zeta5^3 + 2*zeta5^2 - 2*zeta5 - 1)*t^3 + (zeta5^3 + 2*zeta5^2 + 3*zeta5 + 3)*t^2 + (-2*zeta5^2 - zeta5)*t - 1")
])
def test_twisted_alexander_torus_knot_nontrivial(p, q, x_values, expected_str):
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q)
    h1 = BranchedCoverHomology(knot, p)
    
    # x_values are characters values on generators x_0, x_1, ...
    # We need to provide them in the nested format.
    char = Character(h1, [[x_values]])
    
    poly = twisted_alexander_torus_knot(knot, char)
    
    K = CyclotomicField(q)
    R = PolynomialRing(K, 't')
    # Make zeta available for string evaluation if needed
    if 'zeta5' in expected_str: K = CyclotomicField(5); zeta5 = K.gen()
    if 'zeta7' in expected_str: K = CyclotomicField(7); zeta7 = K.gen()
    if 'zeta9' in expected_str: K = CyclotomicField(9); zeta9 = K.gen()
    
    expected = R.fraction_field()(expected_str)
    assert poly == expected

def test_twisted_alexander_torus_knot_input_validation():
    knot_2_3 = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1_2_3 = BranchedCoverHomology(knot_2_3, 2)
    char_2_3 = Character(h1_2_3, [[[0]]])
    
    # 1. Not a torus knot (iterated)
    knot_it = GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5)])
    with pytest.raises(ValueError, match="Knot must be a positive torus knot"):
        twisted_alexander_torus_knot(knot_it, Character(BranchedCoverHomology(knot_it, 2), [[[0], []]]))
        
    # 2. Negative torus knot
    knot_neg = GeneralizedAlgebraicKnot.torus_knot(2, 3, sign=-1)
    with pytest.raises(ValueError, match="Knot must be a positive torus knot"):
        twisted_alexander_torus_knot(knot_neg, char_2_3)
        
    # 3. Wrong N
    h1_bad_n = BranchedCoverHomology(knot_2_3, 3)
    with pytest.raises(ValueError, match="Formula requires character on the 2-fold cover"):
        twisted_alexander_torus_knot(knot_2_3, Character(h1_bad_n, [[[0, 0]]]))
        
    # 4. Null character
    with pytest.raises(TypeError):
        twisted_alexander_torus_knot(knot_2_3, None)
        
    # 5. Character on connected sum
    knot_2_5 = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    sum_knot = knot_2_3 + knot_2_5
    h1_sum = BranchedCoverHomology(sum_knot, 2)
    char_sum = Character(h1_sum, [[[0]], [[0]]])
    with pytest.raises(ValueError, match="Knot must be a positive torus knot"):
        twisted_alexander_torus_knot(sum_knot, char_sum)

    # 6. Non-Character object
    with pytest.raises(TypeError, match="Expected a Character object"):
        twisted_alexander_torus_knot(knot_2_3, "not a character")

    # 7. Knot with multiple components (already covered by 5, but here explicit)
    knot_2_3_2_5 = knot_2_3 + knot_2_5
    h1_2_3_2_5 = BranchedCoverHomology(knot_2_3_2_5, 2)
    char_2_3_2_5 = Character(h1_2_3_2_5, [[[0]], [[0]]])
    with pytest.raises(ValueError, match="Knot must be a positive torus knot"):
        twisted_alexander_torus_knot(knot_2_3_2_5, char_2_3_2_5)

    # 8. Knot with 0 summands
    with pytest.raises(ValueError):
        GeneralizedAlgebraicKnot([])

    # 9. Large N
    h1_large = BranchedCoverHomology(knot_2_3, 10)
    # T(2,3) N=10 has factors [3]. Only 1 value needed.
    with pytest.raises(ValueError, match="Formula requires character on the 2-fold cover"):
        twisted_alexander_torus_knot(knot_2_3, Character(h1_large, [[[0]]]))

    # 10. Normal behavior check
    assert twisted_alexander_torus_knot(knot_2_3, char_2_3) is not None

@pytest.mark.parametrize("p, q, x_val", [
    (2, 3, 1/3), (2, 5, 1/5), (3, 2, 1/2), (2, 7, 1/7), (5, 2, 1/2),
    (4, 3, 1/3), (5, 4, 1/4), (3, 5, 2/5), (2, 9, 1/9), (2, 11, 1/11)
])
def test_character_method_twisted_alexander(p, q, x_val):
    knot = GeneralizedAlgebraicKnot.torus_knot(p, q)
    h1 = BranchedCoverHomology(knot, p)
    factors = len(h1.decomposition[0]['layers'][0]['base_factors'])
    char = Character(h1, [[[x_val] * factors]])
    
    poly = char.twisted_alexander_polynomial()
    assert poly is not None
    
def test_character_method_unsupported_edge_cases():
    # 1. Iterated knot
    knot_it = GeneralizedAlgebraicKnot.iterated_torus_knot([(2, 3), (2, 5)])
    h1_it = BranchedCoverHomology(knot_it, 2)
    char_it = Character(h1_it, [[[0], []]])
    with pytest.raises(NotImplementedError, match="currently only implemented for positive torus knots"):
        char_it.twisted_alexander_polynomial()

    # 2. Connected sum
    knot_sum = GeneralizedAlgebraicKnot.torus_knot(2,3) + GeneralizedAlgebraicKnot.torus_knot(2,5)
    h1_sum = BranchedCoverHomology(knot_sum, 2)
    char_sum = Character(h1_sum, [[[0]], [[0]]])
    with pytest.raises(NotImplementedError):
        char_sum.twisted_alexander_polynomial()

    # 3. Negative torus knot
    knot_neg = GeneralizedAlgebraicKnot.torus_knot(2,3, sign=-1)
    h1_neg = BranchedCoverHomology(knot_neg, 2)
    char_neg = Character(h1_neg, [[[0]]])
    with pytest.raises(NotImplementedError):
        char_neg.twisted_alexander_polynomial()

    # 4. Large p, small q
    knot_large = GeneralizedAlgebraicKnot.torus_knot(7, 2)
    h1_large = BranchedCoverHomology(knot_large, 7)
    char_large = Character(h1_large, [[[0]*6]])
    assert char_large.twisted_alexander_polynomial() is not None

    # 5. Non-coprime (should fail at knot creation)
    with pytest.raises(ValueError):
        GeneralizedAlgebraicKnot.torus_knot(2, 4)

    # 6. p=1 or q=1 (should fail at knot creation)
    with pytest.raises(ValueError):
        GeneralizedAlgebraicKnot.torus_knot(1, 3)

    # 7. Character on N != p
    h1_wrong = BranchedCoverHomology(GeneralizedAlgebraicKnot.torus_knot(2,3), 4)
    # T(2,3) N=4 has factors [3]. Only 1 value needed.
    char_wrong = Character(h1_wrong, [[[0]]])
    with pytest.raises(ValueError):
        char_wrong.twisted_alexander_polynomial()

    # 8. Character with non-rational value (Character init check)
    h1 = BranchedCoverHomology(GeneralizedAlgebraicKnot.torus_knot(2,3), 2)
    with pytest.raises(TypeError):
        Character(h1, [[["a"]]])

    # 9. Character value mismatch (Character init check)
    with pytest.raises(ValueError):
        Character(h1, [[[0, 0]]])

    # 10. Normal behavior check
    char_ok = Character(h1, [[[0]]])
    assert char_ok.twisted_alexander_polynomial() is not None
