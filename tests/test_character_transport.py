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
    TorusPatternPhaseOrbit,
    induced_companion_characters,
    torus_pattern_phase_orbit,
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
        [[[QQ(1) / 5], [QQ(1) / 3, QQ(2) / 3]]],
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
    assert isinstance(transport.pattern_phase_orbit, TorusPatternPhaseOrbit)
    assert transport.pattern_phase_orbit.smith_factors == (5,)
    assert transport.pattern_phase_orbit.generator_values == (QQ(1) / 5,)
    assert transport.pattern_phase_orbit.smith_coordinates == (
        (-1,),
        (-4,),
    )
    assert transport.phase_arguments == (QQ(4) / 5, QQ(1) / 5)
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
    assert via_method.phase_arguments == (QQ(4) / 5, QQ(1) / 5)


def test_outer_pattern_phase_api_can_return_the_complete_deck_orbit():
    r"""The public Character method is useful in both branches of the theorem.

    For the four-fold cover of ``T(2,5)``, the homology is ``Z/5``.  In the
    project's fixed Smith basis, the distinguished element and its first
    three translates have coordinates ``-1,-4,-1,1``.  A character taking the
    Smith generator to ``1/5`` therefore gives phases
    ``4/5,1/5,4/5,1/5``.  Induced transport needs only the first ``h=2`` of
    these, whereas the standalone API deliberately exposes the full orbit.
    """

    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    source = Character(
        BranchedCoverHomology(knot, 4),
        [[[QQ(1) / 5]]],
    )

    orbit = source.outer_torus_pattern_phase_orbit()

    assert orbit.cover_degree == 4
    assert orbit.orbit_length == 4
    assert orbit.deck_powers == (0, 1, 2, 3)
    assert orbit.distinguished_element_coordinates == (-1,)
    assert orbit.smith_coordinates == ((-1,), (-4,), (-1,), (1,))
    assert orbit.phase_arguments == (
        QQ(4) / 5,
        QQ(1) / 5,
        QQ(4) / 5,
        QQ(1) / 5,
    )


def test_general_phase_orbit_keeps_free_smith_coordinates_auditable():
    r"""A torsion character vanishes on free homology without hiding it.

    The six-fold branched cover of the trefoil has two free Smith summands in
    the companion-matrix presentation.  The phase orbit records their
    coordinate vectors, but their only admissible torsion-character values
    are zero, so every evaluated phase is exactly zero.
    """

    orbit = torus_pattern_phase_orbit(
        2,
        3,
        6,
        [0, 0],
        orbit_length=2,
    )

    assert orbit.smith_factors == (0, 0)
    assert len(orbit.smith_coordinates) == 2
    assert all(len(vector) == 2 for vector in orbit.smith_coordinates)
    assert orbit.phase_arguments == (0, 0)


@pytest.mark.parametrize(
    "p, q, cover_degree, generator_values, error_type, message",
    [
        (True, 5, 4, [0], TypeError, "p must be an integer"),
        (2, 4, 4, [0], ValueError, "relatively prime"),
        (2, 5, 1, [0], ValueError, "greater than one"),
        (2, 5, 4, [], ValueError, "one entry per nontrivial Smith factor"),
        (2, 5, 4, [0.2], TypeError, "exact rational number"),
        (2, 5, 4, [QQ(1) / 3], ValueError, "Smith factor 5"),
        (
            2,
            3,
            6,
            [QQ(1) / 2, 0],
            ValueError,
            "torsion-free Smith summand",
        ),
    ],
)
def test_general_phase_orbit_rejects_incompatible_public_data(
    p,
    q,
    cover_degree,
    generator_values,
    error_type,
    message,
):
    r"""Reject inputs that cannot define a character on the pattern cover.

    These cases distinguish topological incompatibility--for example a
    ``1/3`` value on ``Z/5``--from inexact numeric input and malformed cover
    metadata.  In particular, a free Smith coordinate is not silently dropped:
    it is accepted only with the zero extension required by ``Character``.
    """

    with pytest.raises(error_type, match=message):
        torus_pattern_phase_orbit(
            p,
            q,
            cover_degree,
            generator_values,
        )


def test_phase_orbit_record_is_frozen_and_checks_its_displayed_evaluation():
    """The diagnostic coordinates and their evaluated phases cannot diverge."""

    orbit = torus_pattern_phase_orbit(2, 5, 4, [QQ(1) / 5])

    with pytest.raises(FrozenInstanceError):
        orbit.phase_arguments = (0, 0, 0, 0)
    with pytest.raises(ValueError, match="do not evaluate"):
        TorusPatternPhaseOrbit(
            p=orbit.p,
            q=orbit.q,
            cover_degree=orbit.cover_degree,
            orbit_length=orbit.orbit_length,
            smith_factors=orbit.smith_factors,
            generator_values=orbit.generator_values,
            smith_coordinates=orbit.smith_coordinates,
            phase_arguments=(0, 0, 0, 0),
        )


@pytest.mark.parametrize(
    "orbit_length, error_type, message",
    [
        (True, TypeError, "orbit_length must be an integer"),
        (0, ValueError, "between one and cover_degree"),
        (5, ValueError, "between one and cover_degree"),
    ],
)
def test_outer_pattern_phase_api_rejects_invalid_orbit_lengths(
    orbit_length,
    error_type,
    message,
):
    """A requested initial orbit segment must be meaningful in the cover."""

    knot = GeneralizedAlgebraicKnot.torus_knot(2, 5)
    source = Character(
        BranchedCoverHomology(knot, 4),
        [[[QQ(1) / 5]]],
    )

    with pytest.raises(error_type, match=message):
        source.outer_torus_pattern_phase_orbit(
            orbit_length=orbit_length,
        )


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
