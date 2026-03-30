import pytest
from sage.all import QQ, ZZ, Integer
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.H1_branched_cover import BranchedCoverHomologyElement
from gaknot.invariants.character import Character

@pytest.mark.parametrize("knot_desc, N, input_values, expected_values", [
    # 1. T(2,3), N=2: Factors [3]. Basic reduction.
    ([(1, [(2, 3)])], 2, [4], [1]),
    # 2. T(2,3) # T(2,5), N=2: Factors [3, 5]. Multi-summand flat.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [4, 7], [1, 2]),
    # 3. T(2,3; 2,5), N=4: Factors [5, 3, 3]. Iterated knot flat.
    ([(1, [(2, 3), (2, 5)])], 4, [6, 4, 2], [1, 1, 2]),
    # 4. T(2,3) # T(2,3), N=2: Factors [3, 3]. Nested summands.
    ([(1, [(2, 3)]), (1, [(2, 3)])], 2, [[[1]], [[2]]], [1, 2]),
    # 5. T(2,3; 2,5), N=4: Factors [5, 3, 3]. Nested iterated.
    ([(1, [(2, 3), (2, 5)])], 4, [[[1], [4, 5]]], [1, 1, 2]),
    # 6. T(2,3), N=3: Factors [2, 2]. Multiplicity from single torus knot.
    ([(1, [(2, 3)])], 3, [5, 6], [1, 0]),
    # 7. T(2,3), N=6: Factors [0, 0]. Free part (no reduction).
    ([(1, [(2, 3)])], 6, [5, -2], [5, -2]),
    # 8. T(2,5), N=5: Factors [2, 2, 2, 2]. Multiplicity from single torus knot.
    ([(1, [(2, 5)])], 5, [1, 2, 3, 4], [1, 0, 1, 0]),
    # 9. T(2,3) # T(2,5) # T(2,7), N=2: Factors [3, 5, 7]. 3 summands.
    ([(1, [(2, 3)]), (1, [(2, 5)]), (1, [(2, 7)])], 2, [1, 1, 1], [1, 1, 1]),
    # 10. T(2,3; 2,5; 2,7), N=4: Factors [7, 5, 5]. 3 layers.
    ([(1, [(2, 3), (2, 5), (2, 7)])], 4, [[[8], [6, 11], []]], [1, 1, 1])
])
def test_homology_element_creation(knot_desc, N, input_values, expected_values):
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    el = h1.element(input_values)
    assert el.values == [Integer(v) for v in expected_values]

@pytest.mark.parametrize("knot_desc, N, v1, v2, scalar, expected_sum, expected_diff, expected_neg, expected_mul", [
    # 1. T(2,3), N=2: Factors [3].
    ([(1, [(2, 3)])], 2, [1], [2], 2, [0], [2], [2], [2]),
    # 2. T(2,3) # T(2,5), N=2: Factors [3, 5].
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [1, 2], [1, 4], 3, [2, 1], [0, 3], [2, 3], [0, 1]),
    # 3. T(2,3), N=3: Factors [2, 2].
    ([(1, [(2, 3)])], 3, [1, 0], [1, 1], 2, [0, 1], [0, 1], [1, 0], [0, 0]),
    # 4. T(2,3), N=6: Factors [0, 0]. Free part.
    ([(1, [(2, 3)])], 6, [1, 2], [3, 4], 5, [4, 6], [-2, -2], [-1, -2], [5, 10]),
    # 5. T(2,3; 2,5), N=4: Factors [5, 3, 3].
    ([(1, [(2, 3), (2, 5)])], 4, [1, 1, 1], [4, 2, 2], 2, [0, 0, 0], [2, 2, 2], [4, 2, 2], [2, 2, 2]),
    # 6. T(3,4), N=3: Factors [4, 4].
    ([(1, [(3, 4)])], 3, [1, 1], [2, 3], 2, [3, 0], [3, 2], [3, 3], [2, 2]),
    # 7. T(2,5), N=5: Factors [2, 2, 2, 2].
    ([(1, [(2, 5)])], 5, [1, 0, 1, 0], [1, 1, 0, 0], 3, [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]),
    # 8. T(2,7), N=2: Factors [7].
    ([(1, [(2, 7)])], 2, [2], [6], 4, [1], [3], [5], [1]),
    # 9. T(2,3) # T(2,3), N=6: Factors [0, 0, 0, 0].
    ([(1, [(2, 3)]), (1, [(2, 3)])], 6, [1, 1, 1, 1], [1, 2, 3, 4], 2, [2, 3, 4, 5], [0, -1, -2, -3], [-1, -1, -1, -1], [2, 2, 2, 2]),
    # 10. T(2,3; 2,5; 2,7), N=2: Factors [7].
    ([(1, [(2, 3), (2, 5), (2, 7)])], 2, [3], [5], 3, [1], [5], [4], [2])
])
def test_homology_element_arithmetic(knot_desc, N, v1, v2, scalar, expected_sum, expected_diff, expected_neg, expected_mul):
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    el1 = h1.element(v1)
    el2 = h1.element(v2)
    
    assert (el1 + el2).values == [Integer(v) for v in expected_sum]
    assert (el1 - el2).values == [Integer(v) for v in expected_diff]
    assert (-el1).values == [Integer(v) for v in expected_neg]
    assert (el1 * scalar).values == [Integer(v) for v in expected_mul]
    assert (scalar * el1).values == [Integer(v) for v in expected_mul]

@pytest.mark.parametrize("knot_desc, N, char_values, element_values, expected_eval", [
    # 1. T(2,3), N=2: Factor [3].
    ([(1, [(2, 3)])], 2, [[[QQ(1)/3]]], [1], QQ(1)/3),
    # 2. T(2,3), N=2: Factor [3].
    ([(1, [(2, 3)])], 2, [[[QQ(2)/3]]], [2], QQ(1)/3), # 4/3 mod 1 = 1/3
    # 3. T(2,3) # T(2,5), N=2: Factors [3, 5].
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [[[QQ(1)/3]], [[QQ(1)/5]]], [1, 1], QQ(8)/15),
    # 4. T(2,3), N=3: Factors [2, 2].
    ([(1, [(2, 3)])], 3, [[[QQ(1)/2, QQ(1)/2]]], [1, 1], 0), # 1/2 + 1/2 = 1 = 0 mod 1
    # 5. T(2,3), N=6: Factors [0, 0].
    ([(1, [(2, 3)])], 6, [[[QQ(1)/10, QQ(3)/10]]], [1, 1], QQ(4)/10),
    # 6. T(2,3; 2,5), N=4: Factors [5, 3, 3].
    ([(1, [(2, 3), (2, 5)])], 4, [[[QQ(2)/5], [QQ(1)/3, QQ(1)/3]]], [1, 1, 1], QQ(2)/5 + QQ(2)/3 - 1), # 2/5+2/3 = 16/15 -> 1/15
    # 7. T(3,4), N=3: Factors [4, 4].
    ([(1, [(3, 4)])], 3, [[[QQ(1)/4, QQ(3)/4]]], [1, 1], 0),
    # 8. T(2,5), N=5: Factors [2, 2, 2, 2].
    ([(1, [(2, 5)])], 5, [[[QQ(1)/2, 0, QQ(1)/2, 0]]], [1, 1, 1, 1], 0),
    # 9. T(2,3) # T(2,3), N=2: Factors [3, 3].
    ([(1, [(2, 3)]), (1, [(2, 3)])], 2, [[[QQ(1)/3]], [[QQ(2)/3]]], [1, 1], 0),
    # 10. T(2,3; 2,5; 2,7), N=4: Factors [7, 5, 5].
    ([(1, [(2, 3), (2, 5), (2, 7)])], 4, [[[QQ(1)/7], [QQ(1)/5, QQ(2)/5], []]], [1, 1, 1], QQ(1)/7 + QQ(3)/5 - 0), # 1/7+3/5 = 26/35
])
def test_character_evaluation(knot_desc, N, char_values, element_values, expected_eval):
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    char = Character(h1, char_values)
    el = h1.element(element_values)
    assert char(el) == expected_eval

@pytest.mark.parametrize("cable_desc, N, expected_factors, nested_values, expected_flat", [
    # 1. T(2,3; 2,5), N=2: d=gcd(2,2)=2. N_inner=1. Factors [5].
    ([(2, 3), (2, 5)], 2, [5], [[[1], []]], [1]),
    # 2. T(2,3; 2,5), N=4: d=gcd(4,2)=2. N_inner=2, mult 2. Factors [5, 3, 3].
    ([(2, 3), (2, 5)], 4, [5, 3, 3], [[[1], [1, 2]]], [1, 1, 2]),
    # 3. T(2,5; 2,3), N=2: d=gcd(2,2)=2. N_inner=1, mult 2. Factors [3].
    ([(2, 5), (2, 3)], 2, [3], [[[1], []]], [1]),
    # 4. T(2,3; 2,5), N=6: d=gcd(6,2)=2. N_inner=3, mult 2. Outer N=6 factors [5].
    # Inner T(2,3) N=6/2=3, mult 2. Factors [2, 2] x 2 -> [2, 2, 2, 2].
    ([(2, 3), (2, 5)], 6, [5, 2, 2, 2, 2], [[[1], [1, 1, 1, 1]]], [1, 1, 1, 1, 1]),
    # 5. T(2,3; 2,3), N=2: Factors [3]. (Inner N=1)
    ([(2, 3), (2, 3)], 2, [3], [[[1], []]], [1]),
    # 6. T(2,3; 2,3), N=4: Outer [3], Inner N=2 mult 2 [3]x2. Total [3, 3, 3].
    ([(2, 3), (2, 3)], 4, [3, 3, 3], [[[1], [1, 2]]], [1, 1, 2]),
    # 7. T(2,3; 2,3), N=6: Outer [0, 0], Inner N=3 mult 2 [2, 2]x2. Total [0, 0, 2, 2, 2, 2].
    ([(2, 3), (2, 3)], 6, [0, 0, 2, 2, 2, 2], [[[5, 6], [1, 1, 1, 1]]], [5, 6, 1, 1, 1, 1]),
    # 8. T(2,3; 2,5; 2,7), N=2: Factors [7].
    ([(2, 3), (2, 5), (2, 7)], 2, [7], [[[1], [], []]], [1]),
    # 9. T(2,3; 2,5; 2,7), N=4: Factors [7, 5, 5].
    ([(2, 3), (2, 5), (2, 7)], 4, [7, 5, 5], [[[1], [2, 3], []]], [1, 2, 3]),
    # 10. T(2,3; 2,3; 2,3), N=4: Outer [3], Mid N=2 mult 2 [3]x2, Inner N=1 mult 4 []. Total [3, 3, 3].
    ([(2, 3), (2, 3), (2, 3)], 4, [3, 3, 3], [[[1], [1, 1], []]], [1, 1, 1])
])
def test_iterated_knot_structure(cable_desc, N, expected_factors, nested_values, expected_flat):
    knot = GeneralizedAlgebraicKnot([(1, cable_desc)])
    h1 = BranchedCoverHomology(knot, N)
    assert h1.all_invariant_factors == [Integer(f) for f in expected_factors]
    el = h1.element(nested_values)
    assert el.values == [Integer(v) for v in expected_flat]

@pytest.mark.parametrize("sum_desc, N, expected_factors, nested_values, expected_flat", [
    # 1. T(2,3) # T(2,3), N=2: [3, 3].
    ([(1, [(2, 3)]), (1, [(2, 3)])], 2, [3, 3], [[[1]], [[2]]], [1, 2]),
    # 2. T(2,3) # T(2,5), N=2: [3, 5].
    ([(1, [(2, 3)]), (1, [(2, 5)])], 2, [3, 5], [[[1]], [[2]]], [1, 2]),
    # 3. T(2,3) # T(2,3) # T(2,3), N=2: [3, 3, 3].
    ([(1, [(2, 3)]), (1, [(2, 3)]), (1, [(2, 3)])], 2, [3, 3, 3], [[[1]], [[1]], [[1]]], [1, 1, 1]),
    # 4. T(2,3; 2,5) # T(2,3), N=4: [5, 3, 3, 3].
    ([(1, [(2, 3), (2, 5)]), (1, [(2, 3)])], 4, [5, 3, 3, 3], [[[1], [2, 2]], [[1]]], [1, 2, 2, 1]),
    # 5. T(2,3) # T(2,5), N=3: [2, 2, 1]. No, T(2,5) N=3. Delta = t^4-t^3+t^2-t+1.
    # N=3: Res(Delta, t^3-1). t^3-1 = (t-1)(t^2+t+1).
    # Delta(zeta_3) = zeta_3^4-zeta_3^3+zeta_3^2-zeta_3+1 = zeta_3-1+(-zeta_3-1)-zeta_3+1 = -zeta_3-1.
    # (-zeta_3-1)(-zeta_3^2-1) = zeta_3^3 + zeta_3 + zeta_3^2 + 1 = 1 + (-1) + 1 = 1.
    # Factors []. So T(2,5) N=3 has no homology.
    ([(1, [(2, 3)]), (1, [(2, 5)])], 3, [2, 2], [[[1, 1]], [[]]], [1, 1]),
    # 6. T(2,3) # T(2,3), N=3: [2, 2, 2, 2].
    ([(1, [(2, 3)]), (1, [(2, 3)])], 3, [2, 2, 2, 2], [[[1, 1]], [[1, 1]]], [1, 1, 1, 1]),
    # 7. T(2,3) # T(2,3), N=6: [0, 0, 0, 0].
    ([(1, [(2, 3)]), (1, [(2, 3)])], 6, [0, 0, 0, 0], [[[1, 2]], [[3, 4]]], [1, 2, 3, 4]),
    # 8. T(2,5) # T(2,5), N=5: [2,2,2,2, 2,2,2,2].
    ([(1, [(2, 5)]), (1, [(2, 5)])], 5, [2,2,2,2, 2,2,2,2], [[[1,0,1,0]], [[0,1,0,1]]], [1,0,1,0, 0,1,0,1]),
    # 9. T(3,4) # T(2,3), N=3: [4, 4, 2, 2].
    ([(1, [(3, 4)]), (1, [(2, 3)])], 3, [4, 4, 2, 2], [[[1, 1]], [[1, 1]]], [1, 1, 1, 1]),
    # 10. T(2,3; 2,5) # T(2,5; 2,3), N=2: [5, 3].
    ([(1, [(2, 3), (2, 5)]), (1, [(2, 5), (2, 3)])], 2, [5, 3], [[[1], []], [[1], []]], [1, 1])
])
def test_connected_sum_structure(sum_desc, N, expected_factors, nested_values, expected_flat):
    knot = GeneralizedAlgebraicKnot(sum_desc)
    h1 = BranchedCoverHomology(knot, N)
    assert h1.all_invariant_factors == [Integer(f) for f in expected_factors]
    el = h1.element(nested_values)
    assert el.values == [Integer(v) for v in expected_flat]
