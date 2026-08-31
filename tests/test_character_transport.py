r"""Tests for induced companion characters in BCP-II, equation (4.18).

For an ``n``-fold cover of a satellite with winding ``w``, put
``h=gcd(n,w)``.  In the nondivisible branch of Theorem 4.19, the companion
contribution uses ``h`` characters on the ``n/h``-fold branched cover:

``chi_i(v) = chi(t_Q^(i-1) iota_n(v))``.

The homology implementation records these as repeated structural copies.
Successive copy indices use lexicographic order: the outermost index varies
slowest.  The tests below make every copy carry a different exact value, so a
cyclic shift, interleaving, reversal, or accidental factor sorting becomes a
visible failure rather than producing an isomorphic-looking result.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import QQ

from gaknot import (
    BranchedCoverHomology,
    Character,
    GeneralizedAlgebraicKnot,
    InducedCompanionCharacters,
    induced_companion_characters,
)


def test_two_induced_characters_are_the_two_deck_translated_copies():
    r"""The four-cover of a winding-two cable yields two double-cover chars."""

    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    homology = BranchedCoverHomology(knot, 4)

    # The outer T(2,5) pattern contributes Z/5 and occurs once.  Removing it
    # leaves the trefoil in cover degree 4/gcd(4,2)=2; its Z/3 generator occurs
    # in two copies, ordered t_Q^0 iota_4 and t_Q^1 iota_4.
    assert homology.decomposition[0]["layers"] == [
        {
            "cable_index": 1,
            "parameters": (2, 5),
            "effective_N": 4,
            "multiplicity": 1,
            "base_factors": [5],
        },
        {
            "cable_index": 0,
            "parameters": (2, 3),
            "effective_N": 2,
            "multiplicity": 2,
            "base_factors": [3],
        },
    ]
    source = Character(
        homology,
        [[[0], [QQ(1) / 3, QQ(2) / 3]]],
    )

    transport = induced_companion_characters(source)

    assert isinstance(transport, InducedCompanionCharacters)
    assert transport.source_character is source
    assert transport.component_index == 0
    assert transport.cover_degree == 4
    assert transport.winding == 2
    assert transport.h == 2
    assert transport.lower_cover_degree == 2
    assert transport.theorem_indices == (1, 2)
    assert transport.deck_powers == (0, 1)
    assert transport.companion_knot.description == [(1, [(2, 3)])]
    assert transport.companion_homology.cover_degree == 2
    assert [character.values for character in transport] == [
        [QQ(1) / 3],
        [QQ(2) / 3],
    ]
    assert all(
        character.homology is transport.companion_homology
        for character in transport
    )

    # The Character convenience method delegates to the same convention.
    via_method = source.induced_companion_characters()
    assert [character.values for character in via_method] == [
        [QQ(1) / 3],
        [QQ(2) / 3],
    ]


def test_deeper_copy_indices_are_grouped_in_contiguous_lexicographic_blocks():
    r"""An outer h=2 split groups three deeper copies under each chi_i.

    Start in cover degree 12.  The outer winding two produces two companion
    copies in degree six.  Inside each companion, the middle winding three
    produces three copies of the innermost T(2,5) double-cover homology.  The
    six innermost copies must therefore be ordered

    ``(outer 0, inner 0..2), (outer 1, inner 0..2)``.
    """

    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 5), (3, 4), (2, 7)]
    )
    homology = BranchedCoverHomology(knot, 12)
    layers = homology.decomposition[0]["layers"]
    assert [layer["multiplicity"] for layer in layers] == [1, 2, 6]
    assert layers[-1]["base_factors"] == [5]

    # Give each source copy a recognizable, compatible value.  Other layers
    # use zero so this test isolates the grouping of the six inner copies.
    outer_values = [
        0 for _ in range(
            layers[0]["multiplicity"] * len(layers[0]["base_factors"])
        )
    ]
    middle_values = [
        0 for _ in range(
            layers[1]["multiplicity"] * len(layers[1]["base_factors"])
        )
    ]
    inner_values = [
        QQ(1) / 5,
        QQ(2) / 5,
        QQ(3) / 5,
        QQ(4) / 5,
        QQ(0),
        QQ(1) / 5,
    ]
    source = Character(
        homology,
        [[outer_values, middle_values, inner_values]],
    )

    transport = source.induced_companion_characters()

    assert transport.h == 2
    assert transport.lower_cover_degree == 6
    assert transport.companion_knot.description == [
        (1, [(2, 5), (3, 4)])
    ]
    companion_layers = transport.companion_homology.decomposition[0]["layers"]
    assert [layer["multiplicity"] for layer in companion_layers] == [1, 3]

    first, second = transport.characters
    assert first.restrict_to_layer(0, 1) == [
        [QQ(1) / 5],
        [QQ(2) / 5],
        [QQ(3) / 5],
    ]
    assert second.restrict_to_layer(0, 1) == [
        [QQ(4) / 5],
        [QQ(0)],
        [QQ(1) / 5],
    ]


def test_transport_selects_one_connected_sum_component_and_preserves_its_sign():
    """Unrelated components are skipped and negative provenance is retained."""

    unrelated = GeneralizedAlgebraicKnot.torus_knot(2, 3)
    negative_cable = -GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    knot = unrelated + negative_cable
    homology = BranchedCoverHomology(knot, 4)

    # Construct neutral values everywhere, then distinguish the two companion
    # copies of component one.  Component zero must have no influence on the
    # selected transport.
    nested_values = []
    for component in homology.decomposition:
        component_values = []
        for layer in component["layers"]:
            component_values.append([
                0
                for _ in range(
                    layer["multiplicity"] * len(layer["base_factors"])
                )
            ])
        nested_values.append(component_values)
    nested_values[1][1] = [QQ(1) / 3, QQ(2) / 3]
    source = Character(homology, nested_values)

    transport = source.induced_companion_characters(component_index=1)

    assert transport.component_index == 1
    assert transport.companion_knot.description == [(-1, [(2, 3)])]
    assert [character.values for character in transport] == [
        [QQ(1) / 3],
        [QQ(2) / 3],
    ]


@pytest.mark.parametrize("component_index", [-1, 1, 3])
def test_transport_rejects_out_of_range_component_indices(component_index):
    """Structural indices are explicit and do not use negative wraparound."""

    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    source = Character(
        BranchedCoverHomology(knot, 4),
        [[[0], [QQ(1) / 3, QQ(2) / 3]]],
    )

    with pytest.raises(IndexError, match="component_index"):
        source.induced_companion_characters(component_index)


@pytest.mark.parametrize("component_index", [True, 0.0, "0", None])
def test_transport_rejects_nonintegral_component_indices(component_index):
    """Booleans and integer-looking inexact values are not structural IDs."""

    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    source = Character(
        BranchedCoverHomology(knot, 4),
        [[[0], [QQ(1) / 3, QQ(2) / 3]]],
    )

    with pytest.raises(TypeError, match="component_index must be an integer"):
        source.induced_companion_characters(component_index)


def test_transport_requires_an_actual_inner_companion():
    """A one-layer torus pattern leaves only the unrepresented unknot."""

    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    source = Character(
        BranchedCoverHomology(knot, 4),
        [[[QQ(1) / 5]]],
    )

    with pytest.raises(ValueError, match="at least two cabling layers"):
        source.induced_companion_characters()


def test_transport_rejects_the_divisible_ordinary_companion_branch():
    """When n divides w, Theorem 4.19 needs no lower-cover characters."""

    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    source = Character(
        BranchedCoverHomology(knot, 2),
        [[[QQ(1) / 5], []]],
    )

    with pytest.raises(ValueError, match="ordinary companion signatures"):
        source.induced_companion_characters()


def test_transport_factory_rejects_noncharacters():
    """Nested values alone do not identify a parent cover or copy ordering."""

    with pytest.raises(TypeError, match="Character object"):
        induced_companion_characters([[[QQ(1) / 3]]])


def test_transport_record_and_exposed_values_are_immutable_or_defensive():
    """No caller can replace a transported character or mutate its values."""

    knot = GeneralizedAlgebraicKnot.iterated_torus_knot(
        [(2, 3), (2, 5)]
    )
    source = Character(
        BranchedCoverHomology(knot, 4),
        [[[0], [QQ(1) / 3, QQ(2) / 3]]],
    )
    transport = source.induced_companion_characters()

    with pytest.raises(FrozenInstanceError):
        transport.h = 1
    with pytest.raises(FrozenInstanceError):
        transport.characters = ()

    exposed = transport[0].values
    exposed[0] = QQ(0)
    assert transport[0].values == [QQ(1) / 3]
