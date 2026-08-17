import pytest
import warnings
from sage.all import QQ, ZZ
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology, Character

# Suppress DeprecationWarnings from SageMath and other libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*superseded by LazyCombinatorialSpecies.*")
warnings.filterwarnings('ignore', message=".*Importing .* from here is deprecated.*")

@pytest.mark.parametrize("p, q, N, expected_factors, char_values", [
    (2, 3, 2, [3],       [1/3]),          # 1. Trefoil (Z/3)
    (2, 5, 2, [5],       [1/5]),          # 2. T(2,5) (Z/5)
    (2, 7, 2, [7],       [1/7]),          # 3. T(2,7) (Z/7)
    (2, 3, 3, [2, 2],    [1/2, 1/2]),     # 4. T(2,3) at N=3 (Z/2 + Z/2)
    (3, 4, 2, [3],       [1/3]),          # 5. T(3,4) (Z/3)
    (3, 5, 2, [],        []),             # 6. T(3,5) (Trivial H1)
    (2, 5, 4, [5],       [1/5]),          # 7. T(2,5) at N=4 (Z/5)
    (2, 11, 2, [11],     [1/11]),         # 8. T(2,11) (Z/11)
    (2, 31, 2, [31],     [1/31]),         # 9. T(2,31) (Z/31)
    (3, 2, 4, [3],       [1/3])           # 10. T(3,2) at N=4 (Z/3)
])
def test_case_1_simple_torus_parametric(p, q, N, expected_factors, char_values):
    knot = GeneralizedAlgebraicKnot([(1, [(p, q)])])
    homology = BranchedCoverHomology(knot, N)
    
    actual_factors = [f for f in homology.invariant_factors if f > 1]
    assert actual_factors == expected_factors
    
    nested_values = [ [char_values] ]
    chi = Character(homology, nested_values)
    
    layer_vals = chi.restrict_to_layer(0, 0)
    assert layer_vals == [char_values]

@pytest.mark.parametrize("cable, N, outer_factors, inner_factors, inner_values", [
    ([(2, 3), (2, 5)], 3, [], [2, 2], [1/2, 1/2]),
    ([(2, 5), (3, 5)], 2, [], [5], [1/5]),
    ([(3, 4), (3, 5)], 2, [], [3], [1/3]),
    ([(2, 3), (3, 7)], 2, [], [3], [1/3]),
    ([(2, 11), (2, 3)], 3, [2, 2], [], []),
    ([(2, 3), (2, 5)], 5, [2, 2, 2, 2], [], []),
    ([(3, 2), (2, 3)], 2, [3], [], []),
    ([(2, 5), (2, 7)], 2, [7], [], []),
    ([(2, 3), (2, 11)], 2, [11], [], []),
    ([(2, 7), (2, 5)], 3, [], [], [])
])
def test_case_2_degenerate_satellite(cable, N, outer_factors, inner_factors, inner_values):
    knot = GeneralizedAlgebraicKnot([(1, cable)])
    homology = BranchedCoverHomology(knot, N)
    
    assert homology.decomposition[0]["layers"][0]["base_factors"] == outer_factors
    assert homology.decomposition[0]["layers"][1]["base_factors"] == inner_factors
    
    m_outer = homology.decomposition[0]["layers"][0]["multiplicity"]
    m_inner = homology.decomposition[0]["layers"][1]["multiplicity"]
    
    outer_vals_input = [0] * (m_outer * len(outer_factors))
    inner_vals_input = inner_values
    if len(inner_vals_input) != m_inner * len(inner_factors):
        inner_vals_input = inner_values * m_inner
        
    values = [ [ outer_vals_input, inner_vals_input ] ]
    chi = Character(homology, values)
    
    assert chi.restrict_to_layer(0, 0) == [ [0]*len(outer_factors) ] * m_outer
    assert chi.restrict_to_layer(0, 1) == [ inner_values ] * m_inner

@pytest.mark.parametrize("cable, N, outer_factors, inner_factors", [
    ([(2, 5), (3, 2)], 2, [3], [5]),
    ([(2, 3), (3, 2)], 2, [3], [3]),
    ([(2, 3), (5, 2)], 2, [5], [3]),
    ([(2, 7), (3, 2)], 2, [3], [7]),
    ([(2, 5), (5, 2)], 2, [5], [5]),
    ([(2, 3), (7, 2)], 2, [7], [3]),
    ([(2, 11), (3, 2)], 2, [3], [11]),
    ([(3, 2), (5, 2)], 2, [5], [3]),
    ([(5, 2), (3, 2)], 2, [3], [5]),
    ([(2, 5), (7, 2)], 2, [7], [5])
])
def test_case_3_nontrivial_cable_full(cable, N, outer_factors, inner_factors):
    knot = GeneralizedAlgebraicKnot([(1, cable)])
    homology = BranchedCoverHomology(knot, N)
    
    o_vals = [1/f for f in outer_factors]
    i_vals = [1/f for f in inner_factors]
    
    values = [ [ o_vals, i_vals ] ]
    chi = Character(homology, values)
    
    assert chi.restrict_to_layer(0, 0) == [o_vals]
    assert chi.restrict_to_layer(0, 1) == [i_vals]

@pytest.mark.parametrize("p1, q1, p2, q2, N", [
    (5, 2, 3, 2, 2),
    (7, 2, 3, 2, 2),
    (11, 2, 5, 2, 2),
    (3, 2, 5, 2, 2),
    (5, 2, 7, 2, 2),
    (2, 3, 2, 5, 3),
    (2, 5, 2, 3, 2),
    (3, 4, 2, 3, 2),
    (2, 3, 3, 4, 2),
    (2, 11, 2, 3, 2)
])
def test_case_4_connected_sum_geometric_order(p1, q1, p2, q2, N):
    k1 = GeneralizedAlgebraicKnot([(1, [(p1, q1)])])
    k2 = GeneralizedAlgebraicKnot([(1, [(p2, q2)])])
    knot_sum = k1 + k2
    homology = BranchedCoverHomology(knot_sum, N)
    
    f1 = [f for f in BranchedCoverHomology(k1, N).invariant_factors if f > 1]
    f2 = [f for f in BranchedCoverHomology(k2, N).invariant_factors if f > 1]
    
    v1 = [1/f for f in f1]
    v2 = [1/f for f in f2]
    
    values = [ [v1], [v2] ]
    chi = Character(homology, values)
    
    assert chi.restrict_to_layer(0, 0) == [v1]
    assert chi.restrict_to_layer(1, 0) == [v2]

@pytest.mark.parametrize("p, q, N, bad_val", [
    (2, 3, 3, 1/3),
    (2, 5, 2, 1/3),
    (2, 3, 2, 1/2),
    (3, 4, 2, 1/2),
    (2, 7, 2, 1/5),
    (2, 5, 4, 1/2),
    (2, 11, 2, 1/3),
    (2, 3, 3, 1/7),
    (3, 2, 4, 1/2),
    (2, 31, 2, 1/2)
])
def test_validation_modulus_compatibility(p, q, N, bad_val):
    knot = GeneralizedAlgebraicKnot([(1, [(p, q)])])
    homology = BranchedCoverHomology(knot, N)
    factors = [f for f in homology.invariant_factors if f > 1]
    
    if not factors:
        pytest.skip("Trivial homology")
        
    bad_values = [ [ [bad_val] * len(factors) ] ]
    with pytest.raises(ValueError, match="not compatible with Z"):
        Character(homology, bad_values)

@pytest.mark.parametrize("simple_pq, satellite_cable, N", [
    ((2, 5), [(2, 5), (2, 3)], 2),
    ((2, 3), [(2, 3), (2, 5)], 2),
    ((3, 2), [(2, 5), (3, 2)], 2),
    ((2, 7), [(2, 3), (2, 7)], 2),
    ((2, 3), [(2, 5), (2, 3)], 2),
    ((2, 5), [(2, 7), (2, 5)], 2),
    ((2, 11), [(2, 3), (2, 11)], 2),
    ((3, 4), [(2, 3), (3, 4)], 2),
    ((2, 5), [(3, 2), (2, 5)], 2),
    ((2, 3), [(2, 11), (2, 3)], 2)
])
def test_heterogeneous_connected_sum(simple_pq, satellite_cable, N):
    k1 = GeneralizedAlgebraicKnot([(1, [simple_pq])])
    k2 = GeneralizedAlgebraicKnot([(1, satellite_cable)])
    knot_sum = k1 + k2
    homology = BranchedCoverHomology(knot_sum, N)
    
    h1 = BranchedCoverHomology(k1, N)
    h2 = BranchedCoverHomology(k2, N)
    
    v1 = [1/f for f in h1.invariant_factors if f > 1]
    
    v_sat = []
    for layer in h2.decomposition[0]['layers']:
        m = layer['multiplicity']
        f = layer['base_factors']
        v_sat.append([1/x for x in f] * m)
        
    values = [ [v1], v_sat ]
    chi = Character(homology, values)
    
    assert chi.restrict_to_layer(0, 0) == [v1] if v1 else [[]]
    for i, _ in enumerate(v_sat):
        layer_vals = chi.restrict_to_layer(1, i)
        expected = [ [1/x for x in h2.decomposition[0]['layers'][i]['base_factors']] ] * h2.decomposition[0]['layers'][i]['multiplicity']
        assert layer_vals == expected

def test_validation_mixed_validity_satellite():
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3), (3, 2)])])
    homology = BranchedCoverHomology(knot, 2)
    
    valid_values = [ [ [1/3], [1/3] ] ]
    Character(homology, valid_values)
    
    invalid_values = [ [ [1/3], [1/2] ] ]
    with pytest.raises(ValueError, match="not compatible with Z/3Z"):
        Character(homology, invalid_values)

def test_heterogeneous_connected_sum_nontrivial_inner():
    k1 = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    k2 = GeneralizedAlgebraicKnot([(1, [(2, 5), (3, 2)])])
    knot = k1 + k2
    homology = BranchedCoverHomology(knot, 2)
    
    values = [
        [[1/3]],
        [[1/3], [1/5]]
    ]
    chi = Character(homology, values)
    
    assert chi.restrict_to_layer(0, 0) == [[1/3]]
    assert chi.restrict_to_layer(1, 0) == [[1/3]]
    assert chi.restrict_to_layer(1, 1) == [[1/5]]

def test_complex_k3_case():
    desc1 = [(2, 3), (2, 9), (2, 21)]
    desc2 = [(5, 6), (5, 12), (7, 15)]
    knot = GeneralizedAlgebraicKnot([(1, desc1), (-1, desc2)])
    homology = BranchedCoverHomology(knot, 3)
    
    values = []
    for comp in homology.decomposition:
        comp_vals = []
        for layer in comp['layers']:
            m = layer['multiplicity']
            f = layer['base_factors']
            comp_vals.append([1/x if x != 0 else 0 for x in f] * m)
        values.append(comp_vals)
        
    chi = Character(homology, values)
    
    assert len(chi.values) == len(homology.invariant_factors)
    for i, comp in enumerate(homology.decomposition):
        for j, layer in enumerate(comp['layers']):
            m = layer['multiplicity']
            layer_vals = chi.restrict_to_layer(i, j)
            assert len(layer_vals) == m
            for copy_vals in layer_vals:
                assert len(copy_vals) == len(layer['base_factors'])


# The tests below cover the general Character API rather than the distinction
# between torsion and non-torsion homology elements.  Keeping them here makes
# test_torsion_logic.py responsible only for that distinction.


@pytest.fixture
def trefoil_double_cover():
    """Return the homology of the trefoil's double branched cover, Z/3Z."""
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    return BranchedCoverHomology(knot, 2)


@pytest.fixture
def trefoil_character(trefoil_double_cover):
    """Return the character that maps the generator of Z/3Z to 1/3."""
    return Character(trefoil_double_cover, [[[QQ(1) / 3]]])


def test_character_values_are_defensively_copied(trefoil_double_cover):
    # The value 1/3 is valid on the generator of Z/3Z because its order divides
    # three.  In contrast, the replacement value 1/2 used below is invalid.
    char = Character(trefoil_double_cover, [[[QQ(1) / 3]]])

    # Mutating the public value list must affect only the returned copy, not
    # the internal list that passed validation during character construction.
    exported_values = char.values
    exported_values[0] = QQ(1) / 2
    exported_values.append(QQ(0))

    # Both the coordinate count and the original homomorphism remain intact.
    assert char.values == [QQ(1) / 3]
    assert char(trefoil_double_cover.element([1])) == QQ(1) / 3


def test_character_rejects_invalid_homology():
    # Character construction requires the structural decomposition supplied by
    # a BranchedCoverHomology instance; an arbitrary object cannot replace it.
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomology object"):
        Character(None, [[[0]]])


def test_character_rejects_invalid_evaluation_argument(trefoil_character):
    # A raw coordinate list is not enough: evaluation also needs the homology
    # carried by a BranchedCoverHomologyElement.
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomologyElement"):
        trefoil_character([1])


def test_character_rejects_element_from_different_homology(
    trefoil_double_cover, trefoil_character
):
    # Even elements built from the same knot are incompatible when the
    # branched-cover degree, and hence the homology group, is different.
    h2 = BranchedCoverHomology(trefoil_double_cover.knot, 3)
    el2 = h2.element([1, 1])
    with pytest.raises(
        ValueError,
        match="Character and element must belong to the same homology group",
    ):
        trefoil_character(el2)


@pytest.mark.parametrize(
    "char_values, error_message",
    [
        # No component values are supplied for the knot's single summand.
        ([], "Input structure mismatch"),
        # The component is present, but its only satellite layer is omitted.
        ([[]], "Structure mismatch in Component 0"),
        # The layer of a cyclic group has one generator, not two.
        ([[[1, 1]]], "Value mismatch in Component 0, Layer 0"),
    ],
    ids=["summand-count", "layer-count", "value-count"],
)
def test_character_rejects_malformed_value_structure(
    trefoil_double_cover, char_values, error_message
):
    # Each hierarchy level is checked separately so a regression at one level
    # cannot prevent pytest from exercising the other levels.
    with pytest.raises(ValueError, match=error_message):
        Character(trefoil_double_cover, char_values)


def test_character_rejects_non_rational_value(trefoil_double_cover):
    # Character values live in Q/Z, so a string with no rational
    # interpretation must be rejected before any modulus calculation.
    with pytest.raises(TypeError, match="Value must be rational"):
        Character(trefoil_double_cover, [[["invalid"]]])


def test_character_rejects_incompatible_modulus(trefoil_double_cover):
    # A homomorphism Z/3Z -> Q/Z must send a generator to an element whose
    # order divides three; 1/2 has order two and is therefore incompatible.
    with pytest.raises(ValueError, match="is not compatible with Z/3Z"):
        Character(trefoil_double_cover, [[[QQ(1) / 2]]])


@pytest.mark.parametrize(
    "component_index, layer_index, error_message",
    [
        (1, 0, "Component index 1 out of range"),
        (0, 1, "Layer index 1 out of range"),
    ],
    ids=["component-index", "layer-index"],
)
def test_character_rejects_invalid_restriction_index(
    trefoil_character, component_index, layer_index, error_message
):
    # Component and layer bounds are independent contracts and therefore run
    # as separate parametrized cases.
    with pytest.raises(IndexError, match=error_message):
        trefoil_character.restrict_to_layer(component_index, layer_index)
