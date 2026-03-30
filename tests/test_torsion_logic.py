import pytest
from sage.all import QQ, Integer
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.character import Character

@pytest.mark.parametrize("knot_desc, N, char_values, should_fail", [
    # 1. Pure torsion-free: T(2,3), N=6. Factors [0, 0].
    ([(1, [(2, 3)])], 6, [[[QQ(1)/2, 0]]], True),
    # 2. Pure torsion-free: T(2,3), N=6. All zeros should pass.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], False),
    # 3. Pure torsion: T(2,3), N=2. Factor [3].
    ([(1, [(2, 3)])], 2, [[[QQ(1)/3]]], False),
    # 4. Mixed: T(2,5) # T(2,3), N=6. Factors [5, 0, 0].
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1)/5]], [[0.1, 0]]], True),
    # 5. Mixed: T(2,5) # T(2,3), N=6. Correct zeros on free part.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(2)/5]], [[0, 0]]], False),
    # 6. Nested Satellite: T(2,3; 2,3), N=6. Factors [0, 0, 2, 2, 2, 2].
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 1], [0, 0, 0, 0]]], True),
    # 7. Nested Satellite: T(2,3; 2,3), N=6. Correct zeros on free part.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 0], [0.5, 0.5, 0, 0]]], False),
    # 8. Large factors: T(2,11), N=2. Factor [11].
    ([(1, [(2, 11)])], 2, [[[QQ(10)/11]]], False),
    # 9. Concordance inverse: -T(2,3), N=6. Factors [0, 0].
    ([(-1, [(2, 3)])], 6, [[[QQ(1)/3, 0]]], True),
    # 10. Complex connected sum: T(2,3) # -T(2,3), N=6. Factors [0, 0, 0, 0].
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [[[0, 0]], [[0, 0.1]]], True)
])
def test_character_torsion_free_restriction(knot_desc, N, char_values, should_fail):
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    if should_fail:
        with pytest.raises(ValueError, match="Characters must be zero on the torsion-free part"):
            Character(h1, char_values)
    else:
        Character(h1, char_values)

@pytest.mark.parametrize("knot_desc, N, char_values, element_values, should_fail", [
    # 1. T(2,3), N=6. Pure free. el=[1,0] is not torsion.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], [1, 0], True),
    # 2. T(2,3), N=6. Pure free. el=[0,0] is torsion.
    ([(1, [(2, 3)])], 6, [[[0, 0]]], [0, 0], False),
    # 3. T(2,3), N=2. Pure torsion. Any el is torsion.
    ([(1, [(2, 3)])], 2, [[[QQ(1)/3]]], [5], False),
    # 4. T(2,5) # T(2,3), N=6. [5, 0, 0]. el=[1, 0, 0] is torsion.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1)/5]], [[0, 0]]], [1, 0, 0], False),
    # 5. T(2,5) # T(2,3), N=6. [5, 0, 0]. el=[0, 1, 0] is NOT torsion.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1)/5]], [[0, 0]]], [0, 1, 0], True),
    # 6. T(2,5) # T(2,3), N=6. [5, 0, 0]. el=[7, 0, 0] == [2, 0, 0] is torsion.
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [[[QQ(1)/5]], [[0, 0]]], [7, 0, 0], False),
    # 7. T(2,3; 2,3), N=6. [0, 0, 2, 2, 2, 2]. el=[1, 0, 0, 0, 0, 0] is NOT torsion.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 0], [0, 0, 0, 0]]], [1, 0, 0, 0, 0, 0], True),
    # 8. T(2,3; 2,3), N=6. [0, 0, 2, 2, 2, 2]. el=[0, 0, 1, 1, 1, 1] is torsion.
    ([(1, [(2, 3), (2, 3)])], 6, [[[0, 0], [0.5, 0.5, 0.5, 0.5]]], [0, 0, 1, 1, 1, 1], False),
    # 9. T(2,3) # -T(2,3), N=2. Factors [3, 3]. el=[1, 1] is torsion.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 2, [[[QQ(1)/3]], [[QQ(1)/3]]], [1, 1], False),
    # 10. T(2,3) # -T(2,3), N=6. Factors [0, 0, 0, 0]. el=[0,0,0,0] is torsion.
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [[[0,0]], [[0,0]]], [0,0,0,0], False)
])
def test_character_evaluation_torsion_only(knot_desc, N, char_values, element_values, should_fail):
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    char = Character(h1, char_values)
    el = h1.element(element_values)
    if should_fail:
        with pytest.raises(ValueError, match="Character evaluation is only defined for torsion elements"):
            char(el)
    else:
        char(el)

@pytest.mark.parametrize("knot_desc, N, el_values, expected_is_torsion", [
    # 1. T(2,3), N=6. [0, 0].
    ([(1, [(2, 3)])], 6, [0, 0], True),
    # 2. T(2,3), N=6. [1, 0].
    ([(1, [(2, 3)])], 6, [1, 0], False),
    # 3. T(2,3), N=2. [3].
    ([(1, [(2, 3)])], 2, [1], True),
    # 4. T(2,3), N=2. [2].
    ([(1, [(2, 3)])], 2, [2], True),
    # 5. T(2,5) # T(2,3), N=6. [5, 0, 0].
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [3, 0, 0], True),
    # 6. T(2,5) # T(2,3), N=6. [0, 0, 1].
    ([(1, [(2, 5)]), (1, [(2, 3)])], 6, [0, 0, 1], False),
    # 7. T(2,3; 2,3), N=6. [0, 0, 2, 2, 2, 2].
    ([(1, [(2, 3), (2, 3)])], 6, [0, 0, 1, 0, 1, 0], True),
    # 8. T(2,3; 2,3), N=6. [1, 0, 1, 0, 1, 0].
    ([(1, [(2, 3), (2, 3)])], 6, [1, 0, 1, 0, 1, 0], False),
    # 9. T(2,3) # -T(2,3), N=6. [0, 0, 0, 0].
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [0, 1, 0, 0], False),
    # 10. T(2,3) # -T(2,3), N=6. [0, 0, 0, 0].
    ([(1, [(2, 3)]), (-1, [(2, 3)])], 6, [0, 0, 0, 0], True)
])
def test_element_is_torsion(knot_desc, N, el_values, expected_is_torsion):
    knot = GeneralizedAlgebraicKnot(knot_desc)
    h1 = BranchedCoverHomology(knot, N)
    el = h1.element(el_values)
    assert el.is_torsion == expected_is_torsion

def test_character_invalid_input():
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 2)
    
    # 1. Type error for homology
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomology object"):
        Character(None, [[[0]]])
        
    char = Character(h1, [[[QQ(1)/3]]])
    
    # 2. Type error for call
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomologyElement"):
        char([1])
    
    # 3. Value error for different homology
    h2 = BranchedCoverHomology(knot, 3)
    el2 = h2.element([1, 1])
    with pytest.raises(ValueError, match="Character and element must belong to the same homology group"):
        char(el2)
        
    # 4. Input structure mismatch (Summand count)
    with pytest.raises(ValueError, match="Input structure mismatch"):
        Character(h1, [])
        
    # 5. Layer count mismatch
    with pytest.raises(ValueError, match="Structure mismatch in Component 0"):
        Character(h1, [[]])
        
    # 6. Value count mismatch
    with pytest.raises(ValueError, match="Value mismatch in Component 0, Layer 0"):
        Character(h1, [[[1, 1]]])
        
    # 7. Non-rational value
    with pytest.raises(TypeError, match="Value must be rational"):
        Character(h1, [[["invalid"]]])
        
    # 8. Modulus compatibility
    with pytest.raises(ValueError, match="is not compatible with Z/3Z"):
        Character(h1, [[[QQ(1)/2]]])
        
    # 9. Index out of range for restrict_to_layer (component)
    with pytest.raises(IndexError, match="Component index 1 out of range"):
        char.restrict_to_layer(1, 0)
        
    # 10. Index out of range for restrict_to_layer (layer)
    with pytest.raises(IndexError, match="Layer index 1 out of range"):
        char.restrict_to_layer(0, 1)
