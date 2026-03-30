import pytest
from sage.all import QQ, Integer
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology
from gaknot.invariants.character import Character

def test_character_torsion_free_restriction():
    # T(2,3), N=6: Factors [0, 0]. Free part.
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 6)
    
    # Attempting to create a character with non-zero values on the torsion-free part should fail
    with pytest.raises(ValueError, match="Characters must be zero on the torsion-free part"):
        Character(h1, [[[QQ(1)/2, 0]]])
    
    # Character with zero values on torsion-free part should succeed
    char = Character(h1, [[[0, 0]]])
    assert char.values == [0, 0]

def test_character_evaluation_torsion_only():
    # T(2,3), N=6: Factors [0, 0].
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 6)
    
    char = Character(h1, [[[0, 0]]])
    
    # Torsion element (zero in this case)
    el_torsion = h1.element([0, 0])
    assert char(el_torsion) == 0
    
    # Non-torsion element (free part)
    el_free = h1.element([1, 0])
    with pytest.raises(ValueError, match="Character evaluation is only defined for torsion elements"):
        char(el_free)

def test_mixed_homology_torsion_logic():
    # T(2,3) # T(2,3), N=2 and N=6 mixed?
    # Let's use T(2,3) # T(2,3) with N=6 for one and N=2 for another?
    # No, N must be the same for the whole knot.
    
    # Let's use a knot that has both torsion and free parts.
    # T(2,3) with N=6 has only free part [0, 0].
    # T(2,3) with N=2 has only torsion [3].
    
    # A connected sum T(2,3) # T(2,3) where we might have different behaviors?
    # Actually, if we have a knot that results in [3, 0, 0].
    # Let's construct a knot with factors [3, 0, 0].
    # T(2,3) N=2 gives [3].
    # T(2,3) N=6 gives [0, 0].
    # But BranchedCoverHomology takes a knot and ONE N.
    
    # If N=6, T(2,3) gives [0, 0].
    # If N=6, T(2,5) gives [5].
    # So T(2,5) # T(2,3) with N=6 should have factors [5, 0, 0].
    
    knot = GeneralizedAlgebraicKnot([(1, [(2, 5)]), (1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 6)
    assert h1.all_invariant_factors == [Integer(5), Integer(0), Integer(0)]
    
    # Valid character: non-zero on torsion, zero on free
    char = Character(h1, [[[QQ(1)/5]], [[0, 0]]])
    
    # Torsion element: [1, 0, 0]
    el_torsion = h1.element([1, 0, 0])
    assert char(el_torsion) == QQ(1)/5
    
    # Non-torsion element: [0, 1, 0]
    el_free = h1.element([0, 1, 0])
    with pytest.raises(ValueError, match="Character evaluation is only defined for torsion elements"):
        char(el_free)
    
    # Mixed element: [1, 1, 0] -> not torsion
    el_mixed = h1.element([1, 1, 0])
    with pytest.raises(ValueError, match="Character evaluation is only defined for torsion elements"):
        char(el_mixed)

def test_character_invalid_input():
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 2)
    char = Character(h1, [[[QQ(1)/3]]])
    
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomologyElement"):
        char([1])
    
    h2 = BranchedCoverHomology(knot, 3)
    el2 = h2.element([1, 1])
    with pytest.raises(ValueError, match="Character and element must belong to the same homology group"):
        char(el2)
