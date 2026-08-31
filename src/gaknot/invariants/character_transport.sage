"""Transport characters to companion covers in Theorem 4.19.

Let ``P(K, eta)`` be a satellite with nonzero winding number ``w`` and let a
character ``chi`` be defined on the ``n``-fold branched cover.  Put

``h = gcd(n, w)``.

Equations (4.17)--(4.18) of Borodzik--Conway--Politarczyk II identify ``h``
deck-translated copies of ``H_1(Sigma_(n/h)(K))`` inside the satellite cover.
The characters used by the nondivisible branch of Theorem 4.19 are

``chi_i(v) = chi(t_Q^(i-1) iota_n(v)),  i=1,...,h``.

``BranchedCoverHomology`` already records the corresponding direct-sum
decomposition.  Its repeated layer copies use the following lexicographic
convention: the index introduced by an outer satellite stage varies slowest,
and indices introduced by deeper stages vary fastest.  Consequently, if one
companion layer has multiplicity ``r`` in the ``n/h``-fold cover, then its
``h*r`` copies in the satellite cover occur as ``h`` consecutive blocks of
length ``r``.  Restricting the source character to block ``i-1`` in every
inner layer produces ``chi_i`` in the same structural Smith bases.

This module implements precisely that restriction.  It does *not* calculate
the phase values ``chi_i(q_Q(mu_Q^(-w) eta))`` appearing in Theorem 4.19;
those values involve the distinguished pattern element and are separate from
transporting the character on the companion homology.
"""

from dataclasses import dataclass, field

from sage.all import Integer, gcd

from gaknot.core.gaknot import GeneralizedAlgebraicKnot
from gaknot.invariants.H1_branched_cover import BranchedCoverHomology
from gaknot.invariants.character import Character


def _validated_component_index(character, component_index):
    """Return a valid structural component index for ``character``."""

    if isinstance(component_index, bool) or not isinstance(
        component_index,
        (int, Integer),
    ):
        raise TypeError("component_index must be an integer.")
    component_index = int(component_index)
    component_count = len(character.homology.decomposition)
    if component_index < 0 or component_index >= component_count:
        raise IndexError(
            f"component_index {component_index} is outside the range "
            f"0,...,{component_count - 1}."
        )
    return component_index


@dataclass(frozen=True)
class InducedCompanionCharacters:
    r"""The ``h`` companion characters from BCP-II, equation (4.18).

    ``characters[j]`` is the theorem's ``chi_(j+1)`` and therefore restricts
    the source character to the copy
    ``t_Q^j iota_n(H_1(Sigma_(n/h)(K)))``.  Every character has the shared
    parent ``companion_homology``.

    The object implements the sequence operations ``len``, iteration and
    indexing as conveniences, while retaining the cover and winding metadata
    needed to audit how the sequence was obtained.
    """

    source_character: Character = field(repr=False)
    component_index: int
    companion_knot: GeneralizedAlgebraicKnot
    companion_homology: BranchedCoverHomology = field(repr=False)
    cover_degree: object
    winding: object
    h: object
    lower_cover_degree: object
    characters: tuple = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.source_character, Character):
            raise TypeError("source_character must be a Character object.")
        if isinstance(self.component_index, bool) or not isinstance(
            self.component_index,
            (int, Integer),
        ):
            raise TypeError("component_index must be an integer.")
        component_index = int(self.component_index)
        component_count = len(self.source_character.homology.decomposition)
        if component_index < 0 or component_index >= component_count:
            raise IndexError("component_index is outside the source homology.")

        if not isinstance(self.companion_knot, GeneralizedAlgebraicKnot):
            raise TypeError(
                "companion_knot must be a GeneralizedAlgebraicKnot object."
            )
        if not isinstance(self.companion_homology, BranchedCoverHomology):
            raise TypeError(
                "companion_homology must be a BranchedCoverHomology object."
            )

        for name in ("cover_degree", "winding", "h", "lower_cover_degree"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, Integer)):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, Integer(value))

        if self.cover_degree <= 1:
            raise ValueError("cover_degree must be greater than one.")
        if self.winding <= 0:
            raise ValueError("winding must be positive for a GA-knot cable.")
        if self.h != gcd(self.cover_degree, self.winding):
            raise ValueError("h must equal gcd(cover_degree, winding).")
        if self.winding % self.cover_degree == 0:
            raise ValueError(
                "Induced metabelian companion characters belong to the "
                "nondivisible branch of Theorem 4.19."
            )
        if self.lower_cover_degree != self.cover_degree // self.h:
            raise ValueError("lower_cover_degree must equal cover_degree/h.")
        if self.lower_cover_degree <= 1:
            raise ValueError("The lower companion cover must be nontrivial.")

        if self.companion_homology.knot.description != self.companion_knot.description:
            raise ValueError(
                "companion_homology and companion_knot describe different knots."
            )
        if self.companion_homology.cover_degree != self.lower_cover_degree:
            raise ValueError(
                "companion_homology has the wrong lower cover degree."
            )

        if not isinstance(self.characters, tuple):
            raise TypeError("characters must be a tuple.")
        if len(self.characters) != self.h:
            raise ValueError("There must be exactly h induced characters.")
        for character in self.characters:
            if not isinstance(character, Character):
                raise TypeError("characters must contain Character objects.")
            if character.homology is not self.companion_homology:
                raise ValueError(
                    "Every induced character must use companion_homology "
                    "as its parent."
                )

        object.__setattr__(self, "component_index", component_index)

    @property
    def theorem_indices(self):
        """Return the paper's one-based indices ``1,...,h``."""

        return tuple(Integer(index) for index in range(1, int(self.h) + 1))

    @property
    def deck_powers(self):
        """Return the corresponding exponents of ``t_Q``: ``0,...,h-1``."""

        return tuple(Integer(index) for index in range(int(self.h)))

    def __len__(self):
        """Return the number ``h`` of induced characters."""

        return len(self.characters)

    def __iter__(self):
        """Iterate through ``chi_1,...,chi_h`` in theorem order."""

        return iter(self.characters)

    def __getitem__(self, index):
        """Return an induced character by its zero-based Python index."""

        return self.characters[index]


def induced_companion_characters(character, component_index=0):
    r"""Restrict a satellite character to its ``h`` companion-cover copies.

    Args:
        character: A :class:`Character` on the branched-cover homology of a
            GA-knot containing the selected iterated-torus component.
        component_index: The structural connected-sum component whose
            outermost cabling operation is being removed.

    Returns:
        An immutable :class:`InducedCompanionCharacters` record.  Entry ``j``
        is the character ``chi_(j+1)`` from BCP-II, equation (4.18), on the
        ``n/h``-fold cover of the inner companion.

    Raises:
        TypeError: If the public inputs have the wrong types.
        IndexError: If ``component_index`` does not select a component.
        ValueError: If the component has no nontrivial companion or if its
            outer winding is in the divisible branch, where Theorem 4.19 uses
            ordinary rather than metabelian companion forms.
        ArithmeticError: If the source and independently reconstructed
            companion decompositions violate the satellite-copy convention.

    The function supports a selected component of a connected sum and
    preserves its sign in ``companion_knot``.  It performs no signature or
    orientation calculation; the sign is retained solely as structural
    provenance for later phases.
    """

    if not isinstance(character, Character):
        raise TypeError("character must be a Character object.")
    component_index = _validated_component_index(character, component_index)

    source_homology = character.homology
    component = source_homology.decomposition[component_index]
    cable_sequence = [tuple(pair) for pair in component["description"]]
    if len(cable_sequence) < 2:
        raise ValueError(
            "The selected component must have at least two cabling layers "
            "so that removing its outer pattern leaves a companion knot."
        )

    cover_degree = Integer(source_homology.cover_degree)
    winding = Integer(cable_sequence[-1][0])
    h = Integer(gcd(cover_degree, winding))
    if winding % cover_degree == 0:
        raise ValueError(
            "The outer winding is divisible by the cover degree. "
            "Theorem 4.19 uses ordinary companion signatures in this branch, "
            "not induced metabelian companion characters."
        )
    lower_cover_degree = cover_degree // h

    # The selected component is at the outer boundary of its own satellite
    # recursion, so its first homology layer must be the outer pattern with
    # the original cover degree and exactly one copy.
    source_layers = component["layers"]
    outer_layer = source_layers[0]
    expected_outer_index = len(cable_sequence) - 1
    if (
        outer_layer["cable_index"] != expected_outer_index
        or tuple(outer_layer["parameters"]) != cable_sequence[-1]
        or outer_layer["effective_N"] != cover_degree
        or outer_layer["multiplicity"] != 1
    ):
        raise ArithmeticError(
            "The source homology does not have the expected outer satellite "
            "layer."
        )

    companion_sequence = cable_sequence[:-1]
    companion_knot = GeneralizedAlgebraicKnot([
        (component["sign"], companion_sequence),
    ])
    companion_homology = BranchedCoverHomology(
        companion_knot,
        lower_cover_degree,
    )
    companion_layers = companion_homology.decomposition[0]["layers"]
    inner_source_layers = source_layers[1:]
    if len(inner_source_layers) != len(companion_layers):
        raise ArithmeticError(
            "Removing the outer pattern produced an unexpected number of "
            "companion layers."
        )

    # Build one nested Character input per theorem index.  Each entry starts
    # with one component and accumulates its layers in outer-to-inner order.
    induced_nested_values = [[] for _ in range(int(h))]

    for companion_layer_index, (source_layer, companion_layer) in enumerate(
        zip(inner_source_layers, companion_layers)
    ):
        # The same torus-pattern calculation is present on both sides.  The
        # source contains h deck-translated copies of every copy that already
        # occurs inside the lower-cover companion.
        if (
            source_layer["cable_index"] != companion_layer["cable_index"]
            or tuple(source_layer["parameters"])
            != tuple(companion_layer["parameters"])
            or source_layer["effective_N"]
            != companion_layer["effective_N"]
            or source_layer["base_factors"]
            != companion_layer["base_factors"]
            or source_layer["multiplicity"]
            != h * companion_layer["multiplicity"]
        ):
            raise ArithmeticError(
                "The source layer does not split into h copies of the "
                f"companion layer at index {companion_layer_index}."
            )

        source_copy_values = character.restrict_to_layer(
            component_index,
            companion_layer_index + 1,
        )
        copies_per_induced_character = int(companion_layer["multiplicity"])
        expected_source_copies = int(h) * copies_per_induced_character
        if len(source_copy_values) != expected_source_copies:
            raise ArithmeticError(
                "Character layer restriction returned the wrong number of "
                "source copies."
            )

        # Outer index i varies slowest.  Select the complete contiguous block
        # belonging to t_Q^i iota_n, then flatten its internal companion copies
        # back to Character's one-list-per-layer constructor spelling.
        for induced_index in range(int(h)):
            block_start = induced_index * copies_per_induced_character
            block_end = block_start + copies_per_induced_character
            selected_copies = source_copy_values[block_start:block_end]
            flattened_values = [
                value
                for copy_values in selected_copies
                for value in copy_values
            ]
            induced_nested_values[induced_index].append(flattened_values)

    characters = tuple(
        Character(companion_homology, [nested_values])
        for nested_values in induced_nested_values
    )
    return InducedCompanionCharacters(
        source_character=character,
        component_index=component_index,
        companion_knot=companion_knot,
        companion_homology=companion_homology,
        cover_degree=cover_degree,
        winding=winding,
        h=h,
        lower_cover_degree=lower_cover_degree,
        characters=characters,
    )
