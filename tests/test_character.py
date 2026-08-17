"""Tests for torsion-supported characters on branched-cover homology.

A :class:`Character` records a homomorphism from the torsion subgroup of
``H_1(Sigma_N(K))`` to ``Q/Z``.  If a structural homology generator belongs to
``Z/mZ``, its image ``x`` is valid precisely when ``m*x`` is integral.  The
implementation stores the canonical representative of that image in the
half-open interval ``[0,1)``.  A factor zero represents a free ``Z`` summand;
characters in this class are restricted to the torsion subgroup and must
therefore be zero in that coordinate.

The most important convention in this module is the shape of character data.
Construction follows the homology hierarchy::

    values[connected_sum_component][satellite_layer][coordinate]

Satellite layers are ordered from the outermost pattern to the innermost
companion.  The coordinate list for one layer is flat: if the layer has
``multiplicity = r`` and ``s`` base factors, it contains ``r*s`` entries,
ordered copy-by-copy.  By contrast, ``restrict_to_layer(component, layer)``
restores the copy boundary and returns a list with shape::

    restricted_values[copy][coordinate_within_one_copy]

Thus complete constructor input ``[[[a,b,a,b]]]`` for one component containing
two copies of a two-generator layer is returned as ``[[a,b], [a,b]]`` when
restricted to that layer.  Empty lists are also meaningful: a satellite layer
remains part of the structural decomposition even when its cover homology
contributes no generators.

The first part of this file exercises that hierarchy for torus knots,
satellites, and connected sums.  The final section covers defensive exposure
and the general validation API.  Torsion-versus-free evaluation behavior is
tested separately in ``test_torsion_logic.py``.
"""

import pytest
import warnings
from sage.all import QQ, ZZ
from gaknot import GeneralizedAlgebraicKnot, BranchedCoverHomology, Character


# Character construction performs several Sage coercions.  Supported Sage
# versions may emit deprecation warnings from compatibility modules during
# those operations; no test in this module treats DeprecationWarning as part of
# the Character API, so keep that third-party noise out of the test output.
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*superseded by LazyCombinatorialSpecies.*")
warnings.filterwarnings('ignore', message=".*Importing .* from here is deprecated.*")


# ---------------------------------------------------------------------------
# Single-layer characters on ordinary torus knots
# ---------------------------------------------------------------------------

# Every knot in this table has one connected-sum component and one satellite
# layer.  Consequently the required constructor shape is simply
# ``[[char_values]]``, while restriction returns ``[char_values]`` because the
# layer has multiplicity one.  The examples cover cyclic, noncyclic, and
# trivial homology, several cover degrees, and exchanged torus parameters.
@pytest.mark.parametrize("p, q, N, expected_factors, char_values", [
    # Double covers of T(2,q) supply the standard cyclic examples Z/qZ.
    (2, 3, 2, [3],       [1/3]),          # 1. Trefoil (Z/3)
    (2, 5, 2, [5],       [1/5]),          # 2. T(2,5) (Z/5)
    (2, 7, 2, [7],       [1/7]),          # 3. T(2,7) (Z/7)
    # The trefoil's triple cover has two independent order-two generators.
    (2, 3, 3, [2, 2],    [1/2, 1/2]),     # 4. T(2,3) at N=3 (Z/2 + Z/2)
    # The construction is not restricted to winding parameter p=2.
    (3, 4, 2, [3],       [1/3]),          # 5. T(3,4) (Z/3)
    # A trivial group still has a component and layer, but zero coordinates.
    (3, 5, 2, [],        []),             # 6. T(3,5) (Trivial H1)
    # Higher-degree covers need not produce more generators.
    (2, 5, 4, [5],       [1/5]),          # 7. T(2,5) at N=4 (Z/5)
    # Larger moduli ensure compatibility is not accidentally hard-coded for
    # only the small examples used elsewhere in the suite.
    (2, 11, 2, [11],     [1/11]),         # 8. T(2,11) (Z/11)
    (2, 31, 2, [31],     [1/31]),         # 9. T(2,31) (Z/31)
    # Exchanging p and q preserves the knot type but exercises input ordering.
    (3, 2, 4, [3],       [1/3])           # 10. T(3,2) at N=4 (Z/3)
])
def test_case_1_simple_torus_parametric(p, q, N, expected_factors, char_values):
    """One-component, one-layer characters preserve their generator values."""
    knot = GeneralizedAlgebraicKnot([(1, [(p, q)])])
    homology = BranchedCoverHomology(knot, N)
    
    # Character coordinates correspond only to nontrivial finite cyclic
    # factors.  These examples contain no free factors; filtering also makes
    # the intended torsion moduli explicit at the point of construction.
    actual_factors = [f for f in homology.invariant_factors if f > 1]
    assert actual_factors == expected_factors
    
    # The two enclosing lists represent the sole component and its sole layer.
    nested_values = [ [char_values] ]
    chi = Character(homology, nested_values)
    
    # Restriction adds a copy level.  Ordinary torus knots occur once, hence a
    # one-element outer list even when ``char_values`` itself is empty.
    layer_vals = chi.restrict_to_layer(0, 0)
    assert layer_vals == [char_values]


# ---------------------------------------------------------------------------
# Two-layer satellites with an empty contribution
# ---------------------------------------------------------------------------

# A two-pair cable description is written [inner, outer], but its homology
# layers—and therefore character values—are ordered [outer, inner].  Every row
# below has at least one layer with no nontrivial homology.  This is important:
# an empty factor list contributes zero coordinates but does not remove the
# layer from the Character input structure.
@pytest.mark.parametrize("cable, N, outer_factors, inner_factors, inner_values", [
    # Outer T(2,5) is trivial at N=3; inner trefoil contributes (Z/2)^2.
    ([(2, 3), (2, 5)], 3, [], [2, 2], [1/2, 1/2]),
    # The following three rows have a trivial outer layer and one cyclic inner
    # contribution, with different torus-knot parameters.
    ([(2, 5), (3, 5)], 2, [], [5], [1/5]),
    ([(3, 4), (3, 5)], 2, [], [3], [1/3]),
    ([(2, 3), (3, 7)], 2, [], [3], [1/3]),
    # Here the roles reverse: the outer layer contributes and the inner cover
    # is trivial.  The empty ``inner_values`` list remains mandatory.
    ([(2, 11), (2, 3)], 3, [2, 2], [], []),
    ([(2, 3), (2, 5)], 5, [2, 2, 2, 2], [], []),
    # At N=2, an even outer winding reduces the companion degree to one, so
    # these inner layers contribute no homology even though copies exist.
    ([(3, 2), (2, 3)], 2, [3], [], []),
    ([(2, 5), (2, 7)], 2, [7], [], []),
    ([(2, 3), (2, 11)], 2, [11], [], []),
    # Both covers are trivial; the test still supplies two empty layer lists.
    ([(2, 7), (2, 5)], 3, [], [], [])
])
def test_case_2_degenerate_satellite(cable, N, outer_factors, inner_factors, inner_values):
    """Empty layers retain their position and repeated layers retain copies."""
    knot = GeneralizedAlgebraicKnot([(1, cable)])
    homology = BranchedCoverHomology(knot, N)
    
    # Index zero is the outer pattern even though it is the last cable pair in
    # the knot description; index one is the inner companion.
    assert homology.decomposition[0]["layers"][0]["base_factors"] == outer_factors
    assert homology.decomposition[0]["layers"][1]["base_factors"] == inner_factors
    
    # Satellite recursion can create several copies of an inner contribution.
    # Constructor input flattens those copies into one list per layer.
    m_outer = homology.decomposition[0]["layers"][0]["multiplicity"]
    m_inner = homology.decomposition[0]["layers"][1]["multiplicity"]
    
    # Zero is compatible with every finite modulus and with an empty/free
    # contribution, making it a neutral choice for the layer not under focus.
    outer_vals_input = [0] * (m_outer * len(outer_factors))
    inner_vals_input = inner_values
    # Fixture data describes one inner copy.  Repeat it exactly when the
    # satellite formula produces more than one copy of that layer.
    if len(inner_vals_input) != m_inner * len(inner_factors):
        inner_vals_input = inner_values * m_inner
        
    # One component contains two layer-coordinate lists in outer-to-inner order.
    values = [ [ outer_vals_input, inner_vals_input ] ]
    chi = Character(homology, values)
    
    # Restriction reverses the flattening across multiplicity: it returns one
    # factor list for each copy, including explicit empty lists when a copied
    # layer has no generators.
    assert chi.restrict_to_layer(0, 0) == [ [0]*len(outer_factors) ] * m_outer
    assert chi.restrict_to_layer(0, 1) == [ inner_values ] * m_inner


# ---------------------------------------------------------------------------
# Two-layer satellites with nontrivial outer and inner contributions
# ---------------------------------------------------------------------------

# All rows use the double cover and an odd outer winding parameter.  Therefore
# gcd(N,p_outer)=1: the inner companion still uses degree two and occurs once.
# Both layers contribute nontrivial cyclic factors, which isolates ordering
# from the empty-layer and multiplicity concerns tested in the previous table.
@pytest.mark.parametrize("cable, N, outer_factors, inner_factors", [
    # In each cable pair [inner, outer], expected factors are deliberately
    # listed [outer, inner] to match the homology decomposition.
    ([(2, 5), (3, 2)], 2, [3], [5]),
    ([(2, 3), (3, 2)], 2, [3], [3]),
    ([(2, 3), (5, 2)], 2, [5], [3]),
    ([(2, 7), (3, 2)], 2, [3], [7]),
    ([(2, 5), (5, 2)], 2, [5], [5]),
    # Larger outer moduli check that layer restriction does not infer order by
    # sorting the factors numerically.
    ([(2, 3), (7, 2)], 2, [7], [3]),
    ([(2, 11), (3, 2)], 2, [3], [11]),
    # The final rows vary both layers and reverse similar-looking descriptions.
    ([(3, 2), (5, 2)], 2, [5], [3]),
    ([(5, 2), (3, 2)], 2, [3], [5]),
    ([(2, 5), (7, 2)], 2, [7], [5])
])
def test_case_3_nontrivial_cable_full(cable, N, outer_factors, inner_factors):
    """Restriction distinguishes two nonempty layers in structural order."""
    knot = GeneralizedAlgebraicKnot([(1, cable)])
    homology = BranchedCoverHomology(knot, N)
    
    # Sending a generator of Z/fZ to 1/f gives a valid character of exact
    # order f.  Building values from the expected factors makes each layer
    # visibly distinguishable whenever its modulus differs from the other.
    o_vals = [1/f for f in outer_factors]
    i_vals = [1/f for f in inner_factors]
    
    # There is one component, with outer values followed by inner values.
    values = [ [ o_vals, i_vals ] ]
    chi = Character(homology, values)
    
    # Multiplicity is one in this table, hence each restriction contains one
    # copy of the corresponding layer-coordinate list.
    assert chi.restrict_to_layer(0, 0) == [o_vals]
    assert chi.restrict_to_layer(0, 1) == [i_vals]


# ---------------------------------------------------------------------------
# Connected sums of ordinary torus knots
# ---------------------------------------------------------------------------

# A connected sum adds an outer component level to the Character input.  This
# table checks that component order follows the geometric summand order
# ``k1 + k2`` rather than a global sorting of cyclic moduli.  Parameter pairs
# and cover degrees are varied so the two components may have different
# numbers and orders of generators.
@pytest.mark.parametrize("p1, q1, p2, q2, N", [
    # Several double-cover pairs make the factor order easy to distinguish.
    (5, 2, 3, 2, 2),
    (7, 2, 3, 2, 2),
    (11, 2, 5, 2, 2),
    (3, 2, 5, 2, 2),
    (5, 2, 7, 2, 2),
    # Higher-degree and conventional p<q descriptions exercise the same rule.
    (2, 3, 2, 5, 3),
    (2, 5, 2, 3, 2),
    (3, 4, 2, 3, 2),
    # Swapping the connected-sum order must swap the component restrictions.
    (2, 3, 3, 4, 2),
    (2, 11, 2, 3, 2)
])
def test_case_4_connected_sum_geometric_order(p1, q1, p2, q2, N):
    """Layer restriction retains the left-to-right connected-sum order."""
    k1 = GeneralizedAlgebraicKnot([(1, [(p1, q1)])])
    k2 = GeneralizedAlgebraicKnot([(1, [(p2, q2)])])
    knot_sum = k1 + k2
    homology = BranchedCoverHomology(knot_sum, N)
    
    # Compute each component's finite factors independently.  For a one-layer
    # torus knot their sorted order is also their complete structural order.
    f1 = [f for f in BranchedCoverHomology(k1, N).invariant_factors if f > 1]
    f2 = [f for f in BranchedCoverHomology(k2, N).invariant_factors if f > 1]
    
    # As above, 1/f is compatible with a generator of order f and makes the
    # modulus recoverable from the character value.
    v1 = [1/f for f in f1]
    v2 = [1/f for f in f2]
    
    # Each connected-sum component has one layer, producing shape
    # [component][single layer][coordinates].
    values = [ [v1], [v2] ]
    chi = Character(homology, values)
    
    # Restricting component zero or one retrieves the corresponding summand,
    # not whichever cyclic factors happen to sort first numerically.
    assert chi.restrict_to_layer(0, 0) == [v1]
    assert chi.restrict_to_layer(1, 0) == [v2]


# ---------------------------------------------------------------------------
# Compatibility between a cyclic modulus and a character value
# ---------------------------------------------------------------------------

# A proposed value x for a generator of Z/mZ defines a homomorphism to Q/Z if
# and only if m*x is an integer.  Each row deliberately chooses a denominator
# incompatible with the finite factors of the specified torus-knot cover.
@pytest.mark.parametrize("p, q, N, bad_val", [
    # The triple trefoil cover has modulus two, so values of orders 3 or 7 fail.
    (2, 3, 3, 1/3),
    # Odd cyclic factors reject values of unrelated prime or even order.
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
    """Construction rejects generator images whose order does not divide m."""
    knot = GeneralizedAlgebraicKnot([(1, [(p, q)])])
    homology = BranchedCoverHomology(knot, N)
    # These are single-layer cases, so the sorted finite factors are sufficient
    # to build the complete coordinate list.
    factors = [f for f in homology.invariant_factors if f > 1]
    
    # The current table is nontrivial throughout.  Retaining this guard keeps
    # future fixture additions from turning an empty coordinate list into a
    # misleading expected failure.
    if not factors:
        pytest.skip("Trivial homology")
        
    # Repeat the same incompatible value for every generator.  The enclosing
    # lists provide the one component and one layer required by the API.
    bad_values = [ [ [bad_val] * len(factors) ] ]
    with pytest.raises(ValueError, match="not compatible with Z"):
        Character(homology, bad_values)


# ---------------------------------------------------------------------------
# Heterogeneous connected sums and deeper structural examples
# ---------------------------------------------------------------------------

# These cases combine a one-layer torus knot with a two-layer satellite.  The
# test builds the satellite coordinates from its actual decomposition because
# outer recursion may change both the number of factors and their multiplicity.
# This checks that a Character can traverse components with different internal
# shapes without flattening away their boundaries.
@pytest.mark.parametrize("simple_pq, satellite_cable, N", [
    # The table varies which satellite layers contribute in the double cover
    # and includes several choices of inner and outer torus parameters.
    ((2, 5), [(2, 5), (2, 3)], 2),
    ((2, 3), [(2, 3), (2, 5)], 2),
    ((3, 2), [(2, 5), (3, 2)], 2),
    ((2, 7), [(2, 3), (2, 7)], 2),
    ((2, 3), [(2, 5), (2, 3)], 2),
    ((2, 5), [(2, 7), (2, 5)], 2),
    # Larger parameters and p>q descriptions guard against assumptions tied
    # to the small T(2,3) and T(2,5) examples.
    ((2, 11), [(2, 3), (2, 11)], 2),
    ((3, 4), [(2, 3), (3, 4)], 2),
    ((2, 5), [(3, 2), (2, 5)], 2),
    ((2, 3), [(2, 11), (2, 3)], 2)
])
def test_heterogeneous_connected_sum(simple_pq, satellite_cable, N):
    """Restrictions preserve components whose numbers of layers differ."""
    # The simple knot becomes component zero; the satellite becomes component
    # one with layers stored outermost first.
    k1 = GeneralizedAlgebraicKnot([(1, [simple_pq])])
    k2 = GeneralizedAlgebraicKnot([(1, satellite_cable)])
    knot_sum = k1 + k2
    homology = BranchedCoverHomology(knot_sum, N)
    
    # Independent homology objects make the coordinate construction readable
    # without relying on global factor sorting in the connected sum.
    h1 = BranchedCoverHomology(k1, N)
    h2 = BranchedCoverHomology(k2, N)
    
    # Component zero has one layer and one value 1/f for each finite factor.
    v1 = [1/f for f in h1.invariant_factors if f > 1]
    
    # Component one may have different multiplicities in its two layers.
    # Constructor input expects all copies of a layer concatenated, hence the
    # multiplication by ``m`` of the one-copy value list.
    v_sat = []
    for layer in h2.decomposition[0]['layers']:
        m = layer['multiplicity']
        f = layer['base_factors']
        v_sat.append([1/x for x in f] * m)
        
    # The outer list has two differently shaped component entries.
    values = [ [v1], v_sat ]
    chi = Character(homology, values)
    
    # Every current simple component is nontrivial and has exactly one copy.
    # The fallback records the empty restriction shape for a possible future
    # table row with trivial homology.
    assert chi.restrict_to_layer(0, 0) == [v1] if v1 else [[]]
    # Restriction splits each satellite layer's flat constructor input back
    # into one list per copy, each containing one entry per base factor.
    for i, _ in enumerate(v_sat):
        layer_vals = chi.restrict_to_layer(1, i)
        expected = [ [1/x for x in h2.decomposition[0]['layers'][i]['base_factors']] ] * h2.decomposition[0]['layers'][i]['multiplicity']
        assert layer_vals == expected


def test_validation_mixed_validity_satellite():
    """An invalid coordinate is rejected even when every other layer is valid."""
    # With odd outer winding p=3 at degree two, both outer T(3,2) and inner
    # T(2,3) contribute Z/3Z and each occurs once.
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3), (3, 2)])])
    homology = BranchedCoverHomology(knot, 2)
    
    # Mapping both generators to 1/3 satisfies 3*x = 0 in Q/Z.
    valid_values = [ [ [1/3], [1/3] ] ]
    Character(homology, valid_values)
    
    # Only the inner coordinate changes.  Its value 1/2 fails compatibility
    # with Z/3Z, demonstrating that validation is applied layer-by-layer rather
    # than inferred from the first valid layer.
    invalid_values = [ [ [1/3], [1/2] ] ]
    with pytest.raises(ValueError, match="not compatible with Z/3Z"):
        Character(homology, invalid_values)


def test_heterogeneous_connected_sum_nontrivial_inner():
    """A connected sum can address both nontrivial layers of one component."""
    # Component zero is an ordinary trefoil.  Component one is a satellite
    # whose outer T(3,2) layer contributes Z/3Z and whose inner T(2,5) layer
    # contributes Z/5Z; odd outer winding keeps inner multiplicity equal to one.
    k1 = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    k2 = GeneralizedAlgebraicKnot([(1, [(2, 5), (3, 2)])])
    knot = k1 + k2
    homology = BranchedCoverHomology(knot, 2)
    
    # Shape: component 0 / its only layer, followed by component 1 / outer and
    # inner layers.  The distinct denominators expose any ordering mistake.
    values = [
        [[1/3]],
        [[1/3], [1/5]]
    ]
    chi = Character(homology, values)
    
    # Each restriction returns the additional one-copy wrapper.
    assert chi.restrict_to_layer(0, 0) == [[1/3]]
    assert chi.restrict_to_layer(1, 0) == [[1/3]]
    assert chi.restrict_to_layer(1, 1) == [[1/5]]


def test_complex_k3_case():
    """A generated character respects two components with three layers each."""
    # This deliberately large structural example uses two signed, three-layer
    # iterated torus knots and a triple cover.  The sign affects the knot label
    # but not the homology-coordinate organization used by Character.
    desc1 = [(2, 3), (2, 9), (2, 21)]
    desc2 = [(5, 6), (5, 12), (7, 15)]
    knot = GeneralizedAlgebraicKnot([(1, desc1), (-1, desc2)])
    homology = BranchedCoverHomology(knot, 3)
    
    # Generate valid input directly from every structural layer.  For a finite
    # factor m, 1/m is compatible by construction.  A zero factor denotes a
    # free summand and must receive zero because Character models only the
    # torsion-supported homomorphism.  List multiplication flattens all copies
    # into the layer-coordinate list required by the constructor.
    values = []
    for comp in homology.decomposition:
        comp_vals = []
        for layer in comp['layers']:
            m = layer['multiplicity']
            f = layer['base_factors']
            comp_vals.append([1/x if x != 0 else 0 for x in f] * m)
        values.append(comp_vals)
        
    chi = Character(homology, values)
    
    # Flattening must neither lose nor invent coordinates.  Sorting performed
    # by ``invariant_factors`` is irrelevant here because only the count is
    # compared; restriction below checks the structural placement.
    assert len(chi.values) == len(homology.invariant_factors)
    for i, comp in enumerate(homology.decomposition):
        for j, layer in enumerate(comp['layers']):
            m = layer['multiplicity']
            layer_vals = chi.restrict_to_layer(i, j)
            # The outer length recovers the layer multiplicity.
            assert len(layer_vals) == m
            for copy_vals in layer_vals:
                # The inner length recovers the number of factors in one copy,
                # including zero when that layer contributes trivial homology.
                assert len(copy_vals) == len(layer['base_factors'])


# ---------------------------------------------------------------------------
# General Character API, defensive exposure, and malformed inputs
# ---------------------------------------------------------------------------

# The tests below cover the general Character API rather than the distinction
# between torsion and non-torsion homology elements.  Keeping them here makes
# ``test_torsion_logic.py`` responsible only for that distinction.


@pytest.fixture
def trefoil_double_cover():
    """Return the homology of the trefoil's double branched cover, ``Z/3Z``.

    This is the smallest nontrivial parent group in the suite and gives the
    validation tests one component, one layer, and one coordinate.  A failure
    can therefore be attributed to the API contract under test rather than to
    satellite recursion.
    """
    knot = GeneralizedAlgebraicKnot([(1, [(2, 3)])])
    return BranchedCoverHomology(knot, 2)


@pytest.fixture
def trefoil_character(trefoil_double_cover):
    """Return the character mapping the generator of ``Z/3Z`` to ``1/3``.

    Exact Sage rationals are used in the API-focused tests so their purpose is
    independent of any implicit numeric coercion performed by ``Character``.
    """
    return Character(trefoil_double_cover, [[[QQ(1) / 3]]])


def test_character_values_are_defensively_copied(trefoil_double_cover):
    """The public flat value list cannot mutate the validated character."""
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
    """Character construction requires a branched-cover homology parent."""
    # Character construction requires the structural decomposition supplied by
    # a BranchedCoverHomology instance; an arbitrary object cannot replace it.
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomology object"):
        Character(None, [[[0]]])


def test_character_rejects_invalid_evaluation_argument(trefoil_character):
    """Evaluation accepts homology elements rather than raw coordinates."""
    # A raw coordinate list is not enough: evaluation also needs the homology
    # carried by a BranchedCoverHomologyElement.
    with pytest.raises(TypeError, match="Expected a BranchedCoverHomologyElement"):
        trefoil_character([1])


def test_character_rejects_element_from_different_homology(
    trefoil_double_cover, trefoil_character
):
    """Coordinates from another cover cannot be paired with this character."""
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
    """Each component, layer, and coordinate count is validated separately."""
    # Each hierarchy level is checked separately so a regression at one level
    # cannot prevent pytest from exercising the other levels.
    with pytest.raises(ValueError, match=error_message):
        Character(trefoil_double_cover, char_values)


def test_character_rejects_non_rational_value(trefoil_double_cover):
    """Values without a rational interpretation are rejected explicitly."""
    # Character values live in Q/Z, so a string with no rational
    # interpretation must be rejected before any modulus calculation.
    with pytest.raises(TypeError, match="Value must be rational"):
        Character(trefoil_double_cover, [[["invalid"]]])


def test_character_rejects_incompatible_modulus(trefoil_double_cover):
    """A rational value must define a homomorphism on its cyclic generator."""
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
    """Layer restriction reports component and layer bounds independently."""
    # Component and layer bounds are independent contracts and therefore run
    # as separate parametrized cases.
    with pytest.raises(IndexError, match=error_message):
        trefoil_character.restrict_to_layer(component_index, layer_index)
