import pytest
import warnings
from sage.all import ZZ, gcd
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology

# Suppress DeprecationWarnings from SageMath and other libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*superseded by LazyCombinatorialSpecies.*")
warnings.filterwarnings('ignore', message=".*Importing .* from here is deprecated.*")

@pytest.mark.parametrize("p, q, n, expected_factors, expected_str", [
    (2, 3, 2, [3], "(Z/3Z)[T(2,3)]"),
    (2, 3, 3, [2, 2], "(Z/2Z ⊕ Z/2Z)[T(2,3)]"),
    (2, 5, 2, [5], "(Z/5Z)[T(2,5)]"),
    (2, 5, 4, [5], "(Z/5Z)[T(2,5)]"),
    (3, 4, 2, [3], "(Z/3Z)[T(3,4)]"),
    (3, 4, 3, [4, 4], "(Z/4Z ⊕ Z/4Z)[T(3,4)]"),
    (2, 7, 2, [7], "(Z/7Z)[T(2,7)]"),
    (3, 5, 2, [], "0"),
    (2, 3, 6, [0, 0], "(Z ⊕ Z)[T(2,3)]"),
    (5, 2, 2, [5], "(Z/5Z)[T(5,2)]")
])
def test_h1_torus_knot_parametric(p, q, n, expected_factors, expected_str):
    knot = GeneralizedAlgebraicKnot([(1, [(p, q)])])
    h1 = BranchedCoverHomology(knot, n)
    assert len(h1) == 1
    comp = h1[0]
    factors = [f for f in comp['layers'][0]['base_factors'] if f != 1]
    expected_non_trivial = [f for f in expected_factors if f != 1]
    assert factors == expected_non_trivial
    assert str(h1) == expected_str

@pytest.mark.parametrize("desc, n, expected_layers, expected_inv_factors", [
    ([(2, 3), (2, 5)], 2, 
     [{'params': (2, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [5]),
    ([(2, 3), (6, 5)], 2,
     [{'params': (6, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [5]),
    ([(2, 3), (2, 5)], 3,
     [{'params': (2, 5), 'base': [], 'mult': 1}, {'params': (2, 3), 'base': [2, 2], 'mult': 1}], [2, 2]),
    ([(2, 5), (2, 3)], 2,
     [{'params': (2, 3), 'base': [3], 'mult': 1}, {'params': (2, 5), 'base': [], 'mult': 2}], [3]),
    ([(3, 4), (2, 3)], 2,
     [{'params': (2, 3), 'base': [3], 'mult': 1}, {'params': (3, 4), 'base': [], 'mult': 2}], [3]),
    ([(2, 3), (2, 7)], 2,
     [{'params': (2, 7), 'base': [7], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [7]),
    ([(2, 3), (6, 7)], 2,
     [{'params': (6, 7), 'base': [7], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [7]),
    ([(3, 2), (2, 3)], 2,
     [{'params': (2, 3), 'base': [3], 'mult': 1}, {'params': (3, 2), 'base': [], 'mult': 2}], [3]),
    ([(2, 3), (2, 5), (2, 7)], 2,
     [{'params': (2, 7), 'base': [7], 'mult': 1}, {'params': (2, 5), 'base': [], 'mult': 2}, {'params': (2, 3), 'base': [], 'mult': 2}], [7]),
    ([(2, 3), (2, 9)], 2,
     [{'params': (2, 9), 'base': [9], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [9])
])
def test_h1_iterated_torus_knot_parametric(desc, n, expected_layers, expected_inv_factors):
    knot = GeneralizedAlgebraicKnot([(1, desc)])
    h1 = BranchedCoverHomology(knot, n)
    comp = h1[0]
    layers = comp['layers']
    
    for i, exp in enumerate(expected_layers):
        assert layers[i]['parameters'] == exp['params']
        assert [f for f in layers[i]['base_factors'] if f != 1] == [f for f in exp['base'] if f != 1]
        assert layers[i]['multiplicity'] == exp['mult']
    
    assert [f for f in h1.invariant_factors if f != 1] == sorted([f for f in expected_inv_factors if f != 1])

@pytest.mark.parametrize("sum_desc, n, expected_len, expected_signs", [
    ([(1, [(2, 3)]), (-1, [(2, 3), (2, 5)])], 2, 2, [1, -1]),
    ([(1, [(2, 3)]), (1, [(3, 4)])], 2, 2, [1, 1]),
    ([(-1, [(2, 5)]), (-1, [(2, 7)])], 2, 2, [-1, -1]),
    ([(1, [(2, 3)]), (1, [(2, 3)]), (1, [(2, 3)])], 2, 3, [1, 1, 1]),
    ([(1, [(2, 3), (6, 5)]), (-1, [(2, 3)])], 2, 2, [1, -1]),
    ([(1, [(2, 3)]), (-1, [(3, 4)]), (1, [(2, 5)])], 2, 3, [1, -1, 1]),
    ([(1, [(3, 2)]), (1, [(2, 3)])], 2, 2, [1, 1]),
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 2, 2, [1, -1]),
    ([(1, [(2, 3), (2, 5), (2, 7)]), (1, [(3, 4)])], 2, 2, [1, 1]),
    ([(1, [(5, 7)]), (-1, [(5, 7)])], 2, 2, [1, -1])
])
def test_h1_connected_sum_parametric(sum_desc, n, expected_len, expected_signs):
    knot = GeneralizedAlgebraicKnot(sum_desc)
    h1 = BranchedCoverHomology(knot, n)
    assert len(h1) == expected_len
    for i, sign in enumerate(expected_signs):
        assert h1[i]['sign'] == sign

@pytest.mark.parametrize("desc1, desc2, n", [
    ([(1, [(2, 3)])], [(-1, [(2, 3), (2, 5)])], 2),
    ([(1, [(2, 5)])], [(1, [(2, 7)])], 2),
    ([(-1, [(3, 2)])], [(1, [(3, 4)])], 2),
    ([(1, [(2, 3), (6, 5)])], [(-1, [(2, 3)])], 2),
    ([(1, [(2, 3)])], [(1, [(2, 3)])], 3),
    ([(1, [(3, 4)])], [(1, [(4, 5)])], 2),
    ([(1, [(2, 3), (2, 5)])], [(1, [(2, 3), (2, 7)])], 2),
    ([(1, [(5, 2)])], [(-1, [(5, 3)])], 2),
    ([(1, [(2, 3)]), (1, [(3, 4)])], [(1, [(4, 5)])], 2),
    ([(1, [(2, 3)])], [(1, [(3, 4)]), (1, [(4, 5)])], 2)
])
def test_h1_addition_parametric(desc1, desc2, n):
    h1_1 = BranchedCoverHomology(GeneralizedAlgebraicKnot(desc1), n)
    h1_2 = BranchedCoverHomology(GeneralizedAlgebraicKnot(desc2), n)
    h1_sum = h1_1 + h1_2
    
    assert len(h1_sum) == len(h1_1) + len(h1_2)
    for i in range(len(h1_sum)):
        assert h1_sum[i]['index'] == i
