import pytest
from sage.all import Integer, gcd
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*superseded by LazyCombinatorialSpecies.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*Importing .* from here is deprecated.*:DeprecationWarning"
    ),
]


@pytest.mark.parametrize("cover_degree", [2.5, "2", None, True, False])
def test_h1_rejects_non_integer_cover_degree(cover_degree):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    with pytest.raises(TypeError, match="cover degree should be"):
        BranchedCoverHomology(knot, cover_degree)


@pytest.mark.parametrize("cover_degree", [1, 0, -1, Integer(1)])
def test_h1_rejects_cover_degree_below_two(cover_degree):
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    with pytest.raises(ValueError, match="cover degree must be at least two"):
        BranchedCoverHomology(knot, cover_degree)


@pytest.mark.parametrize("knot", [None, "T(2,3)", [], object()])
def test_h1_rejects_invalid_knot(knot):
    with pytest.raises(TypeError, match="knot argument must be"):
        BranchedCoverHomology(knot, 2)


def test_h1_accepts_sage_integer_cover_degree():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, Integer(2))

    assert h1.cover_degree == 2


def test_h1_accepts_knot_subclass():
    class SpecializedKnot(GeneralizedAlgebraicKnot):
        pass

    knot = SpecializedKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 2)

    assert h1.knot is knot


def test_h1_rejects_object_with_matching_type_name():
    impostor_type = type(
        "GeneralizedAlgebraicKnot",
        (),
        {"description": []},
    )

    with pytest.raises(TypeError, match="knot argument must be"):
        BranchedCoverHomology(impostor_type(), 2)


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

    assert h1.knot is knot
    assert h1.cover_degree == n
    assert len(h1) == 1

    component = h1[0]
    assert component['index'] == 0
    assert component['sign'] == 1
    assert component['description'] == [(p, q)]
    assert len(component['layers']) == 1

    layer = component['layers'][0]
    assert layer['cable_index'] == 0
    assert layer['parameters'] == (p, q)
    assert layer['effective_N'] == n
    assert layer['multiplicity'] == 1
    assert layer['base_factors'] == expected_factors

    assert h1.all_invariant_factors == expected_factors
    assert h1.invariant_factors == sorted(expected_factors)
    assert h1.canonical_invariant_factors == expected_factors
    assert h1.betti_number == expected_factors.count(0)
    assert str(h1) == expected_str


def test_h1_negative_knot_string():
    knot = GeneralizedAlgebraicKnot([(-1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 2)

    assert str(h1) == "(Z/3Z)[-T(2,3)]"


def test_h1_repr():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 2)

    assert repr(h1) == "BranchedCoverHomology(knot='T(2,3)', N=2)"


@pytest.mark.parametrize("desc, n, expected_layers, expected_inv_factors", [
    ([(2, 3), (2, 5)], 2, 
     [{'params': (2, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [5]),
    ([(2, 3), (6, 5)], 2,
     [{'params': (6, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [5]),
    ([(2, 3), (2, 5)], 3,
     [{'params': (2, 5), 'base': [], 'mult': 1}, {'params': (2, 3), 'base': [2, 2], 'mult': 1}], [2, 2]),
    ([(2, 3), (2, 5)], 6,
     [{'params': (2, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [2, 2], 'mult': 2}], [2, 2, 2, 2, 5]),
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

    assert h1.knot is knot
    assert h1.cover_degree == n
    assert len(h1) == 1

    component = h1[0]
    assert component['index'] == 0
    assert component['sign'] == 1
    assert component['description'] == desc

    layers = component['layers']
    assert len(layers) == len(expected_layers)

    effective_n = n
    expected_structural_factors = []

    for offset, (layer, expected) in enumerate(zip(layers, expected_layers)):
        assert layer['cable_index'] == len(desc) - offset - 1
        assert layer['parameters'] == expected['params']
        assert layer['effective_N'] == effective_n
        assert layer['multiplicity'] == expected['mult']
        assert layer['base_factors'] == expected['base']

        expected_structural_factors.extend(
            expected['base'] * expected['mult']
        )

        p, _ = expected['params']
        effective_n //= gcd(effective_n, p)

    assert h1.all_invariant_factors == expected_structural_factors
    assert h1.invariant_factors == sorted(expected_inv_factors)
    assert h1.betti_number == expected_structural_factors.count(0)


@pytest.mark.parametrize(
    "sum_desc, n, expected_structural, expected_canonical, expected_str",
    [
        (
            [(1, [(2, 3)]), (-1, [(2, 3), (2, 5)])],
            2,
            [3, 5],
            [15],
            "(Z/3Z)[T(2,3)] ⊕ (Z/5Z)[-T(2,3; 2,5)]",
        ),
        (
            [(1, [(2, 3)]), (-1, [(3, 4)]), (1, [(2, 5)])],
            2,
            [3, 3, 5],
            [3, 15],
            "(Z/3Z)[T(2,3)] ⊕ (Z/3Z)[-T(3,4)] ⊕ (Z/5Z)[T(2,5)]",
        ),
        (
            [(1, [(2, 3), (2, 5), (2, 7)]), (1, [(3, 4)])],
            2,
            [7, 3],
            [21],
            "(Z/7Z)[T(2,3; 2,5; 2,7)] ⊕ (Z/3Z)[T(3,4)]",
        ),
        (
            [(1, [(5, 7)]), (-1, [(5, 7)])],
            2,
            [],
            [],
            "0",
        ),
    ],
)
def test_h1_connected_sum_parametric(
    sum_desc,
    n,
    expected_structural,
    expected_canonical,
    expected_str,
):
    knot = GeneralizedAlgebraicKnot(sum_desc)
    h1 = BranchedCoverHomology(knot, n)

    assert h1.knot is knot
    assert h1.cover_degree == n
    assert len(h1) == len(sum_desc)

    for index, (sign, description) in enumerate(sum_desc):
        component = h1[index]
        expected_component = BranchedCoverHomology(
            GeneralizedAlgebraicKnot([(sign, description)]),
            n,
        )[0]
        expected_component['index'] = index

        assert component == expected_component

    assert h1.all_invariant_factors == expected_structural
    assert h1.invariant_factors == sorted(expected_structural)
    assert h1.canonical_invariant_factors == expected_canonical
    assert str(h1) == expected_str


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
    knot1 = GeneralizedAlgebraicKnot(desc1)
    knot2 = GeneralizedAlgebraicKnot(desc2)
    h1_1 = BranchedCoverHomology(knot1, n)
    h1_2 = BranchedCoverHomology(knot2, n)
    h1_sum = h1_1 + h1_2
    expected = BranchedCoverHomology(knot1 + knot2, n)

    assert h1_sum.knot.description == expected.knot.description
    assert h1_sum.cover_degree == expected.cover_degree
    assert h1_sum.decomposition == expected.decomposition
    assert h1_sum.all_invariant_factors == expected.all_invariant_factors
    assert h1_sum.invariant_factors == expected.invariant_factors
    assert h1_sum.betti_number == expected.betti_number
    assert str(h1_sum) == str(expected)


@pytest.mark.parametrize("other", [None, 3, "not a homology group"])
def test_h1_addition_rejects_non_homology(other):
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    with pytest.raises(TypeError, match="Can only add"):
        h1 + other


def test_h1_addition_rejects_object_with_matching_type_name():
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )
    impostor_type = type(
        "BranchedCoverHomology",
        (),
        {"cover_degree": 2},
    )

    with pytest.raises(TypeError, match="Can only add"):
        h1 + impostor_type()


def test_h1_addition_accepts_subclass():
    class SpecializedHomology(BranchedCoverHomology):
        pass

    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )
    specialized_h1 = SpecializedHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 5),
        2,
    )

    assert (h1 + specialized_h1).all_invariant_factors == [3, 5]


def test_h1_addition_rejects_different_cover_degrees():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    double_cover = BranchedCoverHomology(knot, 2)
    triple_cover = BranchedCoverHomology(knot, 3)

    with pytest.raises(ValueError, match="different cover degrees"):
        double_cover + triple_cover


def test_h1_addition_does_not_alias_operands():
    h1_1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )
    h1_2 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 5),
        2,
    )
    h1_sum = h1_1 + h1_2

    h1_sum.decomposition[0]['layers'][0]['base_factors'].append(99)
    h1_sum[1]['layers'][0]['base_factors'].append(99)

    assert h1_1.all_invariant_factors == [3]
    assert h1_2.all_invariant_factors == [5]
    assert h1_sum.all_invariant_factors == [3, 5]


def test_h1_decomposition_input_is_defensively_copied():
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    original = BranchedCoverHomology(knot, 2)
    decomposition = original.decomposition
    copied = BranchedCoverHomology(knot, 2, decomposition=decomposition)

    decomposition[0]['layers'][0]['base_factors'].append(99)

    assert copied.all_invariant_factors == [3]


def test_h1_index_accepts_sage_integer():
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    assert h1[Integer(0)]['index'] == 0


@pytest.mark.parametrize("index", ["0", 0.5, None, True])
def test_h1_index_rejects_non_integer(index):
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    with pytest.raises(TypeError, match="Summand index must be an integer"):
        h1[index]


@pytest.mark.parametrize("index", [-1, 1, 10])
def test_h1_index_rejects_out_of_range(index):
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    with pytest.raises(IndexError, match="Summand index out of range"):
        h1[index]


@pytest.mark.parametrize("desc, n, expected_structural, expected_canonical", [
    (
        [(1, [(2, 3)]), (1, [(2, 5)])],
        2,
        [3, 5],
        [15],
    ),
    (
        [(1, [(2, 3)]), (1, [(2, 3)])],
        2,
        [3, 3],
        [3, 3],
    ),
    (
        [(1, [(2, 3), (2, 5)])],
        6,
        [5, 2, 2, 2, 2],
        [2, 2, 2, 10],
    ),
    (
        [(1, [(2, 3)])],
        6,
        [0, 0],
        [0, 0],
    ),
])
def test_h1_canonical_invariant_factors(
    desc,
    n,
    expected_structural,
    expected_canonical,
):
    h1 = BranchedCoverHomology(GeneralizedAlgebraicKnot(desc), n)

    assert h1.all_invariant_factors == expected_structural
    assert h1.invariant_factors == sorted(expected_structural)
    assert h1.canonical_invariant_factors == expected_canonical
