"""Tests for first homology of cyclic branched covers.

``BranchedCoverHomology`` records more than the isomorphism type of the
abelian group.  Its decomposition preserves the topology used to construct
the group: connected-sum components contain satellite layers, and each layer
records its effective cover degree and its multiplicity.  Consequently this
module checks two complementary views of the same group:

* ``all_invariant_factors`` keeps structural order (summand, outer-to-inner
  layer, copy, then generator);
* ``canonical_invariant_factors`` forgets that provenance and gives the Smith
  normal form of the complete finitely generated abelian group.

A stored factor ``m > 1`` represents ``Z/mZ``.  A factor ``0`` represents a
free ``Z`` summand, while Smith factors equal to ``1`` are discarded because
they represent the trivial group.  These conventions explain apparently
unusual examples such as ``[3, 5]`` structurally but ``[15]`` canonically, or
``[0, 0]`` for a rank-two free group.

For an iterated torus knot, the cable description is written from the
innermost companion to the outermost pattern.  Homology layers are stored in
the opposite order because the satellite formula is evaluated from outside
in.  If a layer has winding parameter ``p`` and current cover degree ``N``,
then ``d = gcd(N, p)``; the next inner layer uses degree ``N/d`` and occurs in
``d`` times as many copies.  Several tests below verify this recursion rather
than checking only its final flattened output.
"""

import pytest
from sage.all import Integer, gcd
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology


# Sage imports deprecated compatibility modules in some supported versions.
# Those third-party warnings are unrelated to the homology behavior under
# test, so suppress only their specific messages rather than all warnings.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*superseded by LazyCombinatorialSpecies.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*Importing .* from here is deprecated.*:DeprecationWarning"
    ),
]


# ---------------------------------------------------------------------------
# Constructor validation and Python/Sage type boundaries
# ---------------------------------------------------------------------------

# ``bool`` deserves explicit coverage: it is a subclass of ``int`` in Python,
# but ``True`` and ``False`` are not meaningful cover degrees.
@pytest.mark.parametrize("cover_degree", [2.5, "2", None, True, False])
def test_h1_rejects_non_integer_cover_degree(cover_degree):
    """The cover degree must be an actual Python or Sage integer."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    with pytest.raises(TypeError, match="cover degree should be"):
        BranchedCoverHomology(knot, cover_degree)


@pytest.mark.parametrize("cover_degree", [1, 0, -1, Integer(1)])
def test_h1_rejects_cover_degree_below_two(cover_degree):
    """Degree one and nonpositive degrees do not define the supported covers."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)

    with pytest.raises(ValueError, match="cover degree must be at least two"):
        BranchedCoverHomology(knot, cover_degree)


@pytest.mark.parametrize("knot", [None, "T(2,3)", [], object()])
def test_h1_rejects_invalid_knot(knot):
    """Arbitrary descriptions and objects must not masquerade as knot models."""
    with pytest.raises(TypeError, match="knot argument must be"):
        BranchedCoverHomology(knot, 2)


def test_h1_accepts_sage_integer_cover_degree():
    """Sage ``Integer`` values are first-class inputs throughout the package."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, Integer(2))

    assert h1.cover_degree == 2


def test_h1_accepts_knot_subclass():
    """Validation uses inheritance, so specialized knot implementations work."""
    class SpecializedKnot(GeneralizedAlgebraicKnot):
        pass

    knot = SpecializedKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 2)

    assert h1.knot is knot


def test_h1_rejects_object_with_matching_type_name():
    """A matching class name is insufficient without the required inheritance."""
    # This object resembles the old style of name-based type spoofing: it has
    # the expected class name and a plausible attribute, but no knot behavior.
    impostor_type = type(
        "GeneralizedAlgebraicKnot",
        (),
        {"description": []},
    )

    with pytest.raises(TypeError, match="knot argument must be"):
        BranchedCoverHomology(impostor_type(), 2)


# ---------------------------------------------------------------------------
# One-layer decompositions: cyclic covers of ordinary torus knots
# ---------------------------------------------------------------------------

# The table covers finite cyclic groups, repeated cyclic factors, trivial
# homology, a free group, and symmetry under exchanging the torus parameters.
@pytest.mark.parametrize("p, q, n, expected_factors, expected_str", [
    # The familiar double branched cover of the trefoil has H_1 = Z/3Z.
    (2, 3, 2, [3], "(Z/3Z)[T(2,3)]"),
    # The triple cover has two independent generators, both of order two.
    (2, 3, 3, [2, 2], "(Z/2Z ⊕ Z/2Z)[T(2,3)]"),
    # Different covers can yield the same abstract homology group.
    (2, 5, 2, [5], "(Z/5Z)[T(2,5)]"),
    (2, 5, 4, [5], "(Z/5Z)[T(2,5)]"),
    # Cases with p > 2 exercise the general companion-matrix calculation.
    (3, 4, 2, [3], "(Z/3Z)[T(3,4)]"),
    (3, 4, 3, [4, 4], "(Z/4Z ⊕ Z/4Z)[T(3,4)]"),
    (2, 7, 2, [7], "(Z/7Z)[T(2,7)]"),
    # An empty factor list denotes the trivial group, not a missing result.
    (3, 5, 2, [], "0"),
    # Zero Smith entries record a rank-two free part in the six-fold cover.
    (2, 3, 6, [0, 0], "(Z ⊕ Z)[T(2,3)]"),
    # T(p,q) and T(q,p) are isotopic, but the original label is preserved.
    (5, 2, 2, [5], "(Z/5Z)[T(5,2)]")
])
def test_h1_torus_knot_parametric(p, q, n, expected_factors, expected_str):
    """A torus knot produces one component containing exactly one layer."""
    knot = GeneralizedAlgebraicKnot([(1, [(p, q)])])
    h1 = BranchedCoverHomology(knot, n)

    # Construction retains the precise knot object and the requested degree;
    # ``len`` counts connected-sum components, not cyclic generators.
    assert h1.knot is knot
    assert h1.cover_degree == n
    assert len(h1) == 1

    # Even an ordinary torus knot uses the general component/layer schema.
    component = h1[0]
    assert component['index'] == 0
    assert component['sign'] == 1
    assert component['description'] == [(p, q)]
    assert len(component['layers']) == 1

    # With no satellite recursion, the sole layer uses the original degree,
    # occurs once, and contributes the complete torus-knot computation.
    layer = component['layers'][0]
    assert layer['cable_index'] == 0
    assert layer['parameters'] == (p, q)
    assert layer['effective_N'] == n
    assert layer['multiplicity'] == 1
    assert layer['base_factors'] == expected_factors

    # In these one-layer examples, structural and canonical factors coincide.
    # Sorting is still checked separately because ``invariant_factors`` is the
    # legacy, sorted structural view exposed by the public API.
    assert h1.all_invariant_factors == expected_factors
    assert h1.invariant_factors == sorted(expected_factors)
    assert h1.canonical_invariant_factors == expected_factors
    # Each zero diagonal entry contributes one generator to the free part.
    assert h1.betti_number == expected_factors.count(0)
    assert str(h1) == expected_str


def test_h1_negative_knot_string():
    """Mirroring changes the component label, not the abstract homology group."""
    knot = GeneralizedAlgebraicKnot([(-1, [(2, 3)])])
    h1 = BranchedCoverHomology(knot, 2)

    assert str(h1) == "(Z/3Z)[-T(2,3)]"


def test_h1_repr():
    """The developer representation identifies both defining inputs."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    h1 = BranchedCoverHomology(knot, 2)

    assert repr(h1) == "BranchedCoverHomology(knot='T(2,3)', N=2)"


# ---------------------------------------------------------------------------
# Iterated torus knots: outer-to-inner satellite recursion
# ---------------------------------------------------------------------------

# ``desc`` is written inner-to-outer, whereas each ``expected_layers`` list is
# written in the decomposition's outer-to-inner order.  ``base`` gives one
# copy of a layer's Smith factors; ``mult`` says how many copies the satellite
# formula contributes.  Thus the factor list is obtained by repeating each
# ``base`` list ``mult`` times while walking through the layers.
@pytest.mark.parametrize("desc, n, expected_layers, expected_inv_factors", [
    # gcd(2,2)=2 makes the inner cover degree one, so only outer Z/5Z remains.
    ([(2, 3), (2, 5)], 2, 
     [{'params': (2, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [5]),
    # A different outer winding number with the same gcd has the same recursion.
    ([(2, 3), (6, 5)], 2,
     [{'params': (6, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [5]),
    # gcd(3,2)=1 leaves the inner degree and multiplicity unchanged.
    ([(2, 3), (2, 5)], 3,
     [{'params': (2, 5), 'base': [], 'mult': 1}, {'params': (2, 3), 'base': [2, 2], 'mult': 1}], [2, 2]),
    # At degree six, the inner triple cover appears twice, contributing four 2s.
    ([(2, 3), (2, 5)], 6,
     [{'params': (2, 5), 'base': [5], 'mult': 1}, {'params': (2, 3), 'base': [2, 2], 'mult': 2}], [2, 2, 2, 2, 5]),
    # Reversing the cable sequence changes which torus knot is the outer layer.
    ([(2, 5), (2, 3)], 2,
     [{'params': (2, 3), 'base': [3], 'mult': 1}, {'params': (2, 5), 'base': [], 'mult': 2}], [3]),
    # These examples vary the inner companion while retaining outer T(2,3).
    ([(3, 4), (2, 3)], 2,
     [{'params': (2, 3), 'base': [3], 'mult': 1}, {'params': (3, 4), 'base': [], 'mult': 2}], [3]),
    # The outer pattern can contribute cyclic groups of other odd orders.
    ([(2, 3), (2, 7)], 2,
     [{'params': (2, 7), 'base': [7], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [7]),
    ([(2, 3), (6, 7)], 2,
     [{'params': (6, 7), 'base': [7], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [7]),
    # Parameter order in the inner torus knot does not disturb layer indexing.
    ([(3, 2), (2, 3)], 2,
     [{'params': (2, 3), 'base': [3], 'mult': 1}, {'params': (3, 2), 'base': [], 'mult': 2}], [3]),
    # Three layers ensure recursion continues after an empty middle contribution.
    ([(2, 3), (2, 5), (2, 7)], 2,
     [{'params': (2, 7), 'base': [7], 'mult': 1}, {'params': (2, 5), 'base': [], 'mult': 2}, {'params': (2, 3), 'base': [], 'mult': 2}], [7]),
    # Composite q is allowed when it remains coprime to the winding number p.
    ([(2, 3), (2, 9)], 2,
     [{'params': (2, 9), 'base': [9], 'mult': 1}, {'params': (2, 3), 'base': [], 'mult': 2}], [9])
])
def test_h1_iterated_torus_knot_parametric(desc, n, expected_layers, expected_inv_factors):
    """Every satellite layer retains its parameters, degree, and multiplicity."""
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

    # Reconstruct the structural factor sequence independently from the layer
    # expectations while simultaneously following the effective-degree rule.
    effective_n = n
    expected_structural_factors = []

    for offset, (layer, expected) in enumerate(zip(layers, expected_layers)):
        # Layer zero is the last (outermost) cable pair; subsequent layers move
        # backward through the original inner-to-outer description.
        assert layer['cable_index'] == len(desc) - offset - 1
        assert layer['parameters'] == expected['params']
        assert layer['effective_N'] == effective_n
        assert layer['multiplicity'] == expected['mult']
        assert layer['base_factors'] == expected['base']

        expected_structural_factors.extend(
            expected['base'] * expected['mult']
        )

        # Litherland's satellite formula replaces N by N/gcd(N,p) before
        # computing the next inner companion.
        p, _ = expected['params']
        effective_n //= gcd(effective_n, p)

    assert h1.all_invariant_factors == expected_structural_factors
    # The legacy view discards layer order but retains the same cyclic summands.
    assert h1.invariant_factors == sorted(expected_inv_factors)
    # Free rank is read directly from zero factors, even inside repeated layers.
    assert h1.betti_number == expected_structural_factors.count(0)


# ---------------------------------------------------------------------------
# Connected sums: component provenance versus abstract group structure
# ---------------------------------------------------------------------------

# Homology takes connected sums to direct sums.  The structural factors retain
# the source component of every cyclic group, while the canonical factors may
# combine coprime cyclic summands.  The displayed string intentionally follows
# the structural splitting so that the knot label and sign remain visible.
@pytest.mark.parametrize(
    "sum_desc, n, expected_structural, expected_canonical, expected_str",
    [
        # Z/3 ⊕ Z/5 is cyclic of order 15, although its two summands arise
        # from different knot components and are displayed separately.
        (
            [(1, [(2, 3)]), (-1, [(2, 3), (2, 5)])],
            2,
            [3, 5],
            [15],
            "(Z/3Z)[T(2,3)] ⊕ (Z/5Z)[-T(2,3; 2,5)]",
        ),
        # Z/3 ⊕ Z/3 ⊕ Z/5 has canonical factors 3 | 15.
        (
            [(1, [(2, 3)]), (-1, [(3, 4)]), (1, [(2, 5)])],
            2,
            [3, 3, 5],
            [3, 15],
            "(Z/3Z)[T(2,3)] ⊕ (Z/3Z)[-T(3,4)] ⊕ (Z/5Z)[T(2,5)]",
        ),
        # Coprime factors 7 and 3 similarly combine into one Z/21 summand.
        (
            [(1, [(2, 3), (2, 5), (2, 7)]), (1, [(3, 4)])],
            2,
            [7, 3],
            [21],
            "(Z/7Z)[T(2,3; 2,5; 2,7)] ⊕ (Z/3Z)[T(3,4)]",
        ),
        # Both double covers here have trivial H_1; signs do not alter that.
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
    """Connected-sum components remain individually inspectable and additive."""
    knot = GeneralizedAlgebraicKnot(sum_desc)
    h1 = BranchedCoverHomology(knot, n)

    assert h1.knot is knot
    assert h1.cover_degree == n
    assert len(h1) == len(sum_desc)

    # Compute each component independently.  Equality with the corresponding
    # entry proves that the combined decomposition preserves sign, cable data,
    # layer data, and factor order—not merely the final abstract group.
    for index, (sign, description) in enumerate(sum_desc):
        component = h1[index]
        expected_component = BranchedCoverHomology(
            GeneralizedAlgebraicKnot([(sign, description)]),
            n,
        )[0]
        # A one-summand calculation naturally has index zero; relocate it to
        # the component's position in the connected sum before comparison.
        expected_component['index'] = index

        assert component == expected_component

    assert h1.all_invariant_factors == expected_structural
    # ``invariant_factors`` is historical terminology for the sorted structural
    # list; the genuinely canonical list is exposed by the next property.
    assert h1.invariant_factors == sorted(expected_structural)
    assert h1.canonical_invariant_factors == expected_canonical
    assert str(h1) == expected_str


# ---------------------------------------------------------------------------
# Addition of homology objects
# ---------------------------------------------------------------------------

# Addition should model a direct sum at the object level.  These rows mix
# positive and negative knots, torus and iterated knots, repeated summands, and
# operands that are themselves connected sums.  Rather than duplicating all
# expected dictionaries in the table, the test compares with an independently
# constructed homology object for the connected sum of the underlying knots.
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
    """Object addition agrees with recomputation from the connected-sum knot."""
    knot1 = GeneralizedAlgebraicKnot(desc1)
    knot2 = GeneralizedAlgebraicKnot(desc2)
    h1_1 = BranchedCoverHomology(knot1, n)
    h1_2 = BranchedCoverHomology(knot2, n)
    h1_sum = h1_1 + h1_2
    expected = BranchedCoverHomology(knot1 + knot2, n)

    # Check both the defining metadata and every public representation.  In
    # particular, equality of strings alone would miss malformed layer data.
    assert h1_sum.knot.description == expected.knot.description
    assert h1_sum.cover_degree == expected.cover_degree
    assert h1_sum.decomposition == expected.decomposition
    assert h1_sum.all_invariant_factors == expected.all_invariant_factors
    assert h1_sum.invariant_factors == expected.invariant_factors
    assert h1_sum.betti_number == expected.betti_number
    assert str(h1_sum) == str(expected)


@pytest.mark.parametrize("other", [None, 3, "not a homology group"])
def test_h1_addition_rejects_non_homology(other):
    """Direct sum is defined only between homology objects."""
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    with pytest.raises(TypeError, match="Can only add"):
        h1 + other


def test_h1_addition_rejects_object_with_matching_type_name():
    """Addition also uses inheritance rather than a spoofable class name."""
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )
    # Supply the attribute most likely to be consulted by addition so the
    # failure demonstrates a real type check, not merely a missing attribute.
    impostor_type = type(
        "BranchedCoverHomology",
        (),
        {"cover_degree": 2},
    )

    with pytest.raises(TypeError, match="Can only add"):
        h1 + impostor_type()


def test_h1_addition_accepts_subclass():
    """A specialized homology implementation remains a valid direct summand."""
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
    """Groups from different covering spaces cannot be combined by this API."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    double_cover = BranchedCoverHomology(knot, 2)
    triple_cover = BranchedCoverHomology(knot, 3)

    with pytest.raises(ValueError, match="different cover degrees"):
        double_cover + triple_cover


# ---------------------------------------------------------------------------
# Ownership, defensive copies, and component indexing
# ---------------------------------------------------------------------------

def test_h1_addition_does_not_alias_operands():
    """Public snapshots of a sum cannot mutate either the sum or its operands."""
    h1_1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )
    h1_2 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 5),
        2,
    )
    h1_sum = h1_1 + h1_2

    # Exercise both public routes to nested component data.  ``decomposition``
    # returns the whole hierarchy and indexing returns one component; both must
    # be deep copies because ``base_factors`` is nested several containers down.
    h1_sum.decomposition[0]['layers'][0]['base_factors'].append(99)
    h1_sum[1]['layers'][0]['base_factors'].append(99)

    # No synthetic factor leaks into the original operands or the sum itself.
    assert h1_1.all_invariant_factors == [3]
    assert h1_2.all_invariant_factors == [5]
    assert h1_sum.all_invariant_factors == [3, 5]


def test_h1_decomposition_input_is_defensively_copied():
    """The optional precomputed decomposition remains owned by its caller."""
    knot = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    original = BranchedCoverHomology(knot, 2)
    decomposition = original.decomposition
    copied = BranchedCoverHomology(knot, 2, decomposition=decomposition)

    # This mutation occurs after construction and reaches the deepest mutable
    # list in the caller's copy.  Retaining it internally would add a fake
    # Z/99Z summand to ``copied``.
    decomposition[0]['layers'][0]['base_factors'].append(99)

    assert copied.all_invariant_factors == [3]


def test_h1_index_accepts_sage_integer():
    """Component access accepts Sage integers just as construction does."""
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    assert h1[Integer(0)]['index'] == 0


@pytest.mark.parametrize("index", ["0", 0.5, None, True])
def test_h1_index_rejects_non_integer(index):
    """Component indices exclude coercible values and Python booleans."""
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    with pytest.raises(TypeError, match="Summand index must be an integer"):
        h1[index]


@pytest.mark.parametrize("index", [-1, 1, 10])
def test_h1_index_rejects_out_of_range(index):
    """Indexing follows component identifiers rather than Python wraparound."""
    h1 = BranchedCoverHomology(
        GeneralizedAlgebraicKnot.torus_knot(2, 3),
        2,
    )

    with pytest.raises(IndexError, match="Summand index out of range"):
        h1[index]


# ---------------------------------------------------------------------------
# Canonical invariant-factor decomposition
# ---------------------------------------------------------------------------

# A canonical invariant-factor sequence d_1, ..., d_r satisfies
# d_1 | d_2 | ... | d_r.  Obtaining it can merge prime-power information from
# different structural cyclic summands.  These examples distinguish that
# algebraic normalization from the ordering-only behavior of
# ``invariant_factors``.
@pytest.mark.parametrize("desc, n, expected_structural, expected_canonical", [
    # Chinese remainder theorem: Z/3 ⊕ Z/5 is isomorphic to Z/15.
    (
        [(1, [(2, 3)]), (1, [(2, 5)])],
        2,
        [3, 5],
        [15],
    ),
    # Equal factors already obey the divisibility condition and remain split.
    (
        [(1, [(2, 3)]), (1, [(2, 3)])],
        2,
        [3, 3],
        [3, 3],
    ),
    # The 5-primary contribution joins one 2-primary factor to give 10; the
    # remaining three order-two factors precede it in the divisibility chain.
    (
        [(1, [(2, 3), (2, 5)])],
        6,
        [5, 2, 2, 2, 2],
        [2, 2, 2, 10],
    ),
    # Zero factors denote infinite cyclic summands and are retained verbatim.
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
    """Canonicalization changes presentation without changing the group."""
    h1 = BranchedCoverHomology(GeneralizedAlgebraicKnot(desc), n)

    # Structural order is meaningful provenance.  The legacy property merely
    # sorts it, whereas canonicalization computes a new Smith normal form.
    assert h1.all_invariant_factors == expected_structural
    assert h1.invariant_factors == sorted(expected_structural)
    assert h1.canonical_invariant_factors == expected_canonical
