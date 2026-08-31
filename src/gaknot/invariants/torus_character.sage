#!/usr/bin/env sage -python

r"""Character orbits for the ``p``-fold cover of a torus knot ``T(p,q)``.

Several formulas used by :mod:`gaknot` encode a character on

``H_1(Sigma_p(T(p,q))) = (ZZ/qZZ)^(p-1)``

by a cyclic sequence ``a_0, ..., a_{p-1}`` in ``ZZ/qZZ``.  The sequence is
obtained by choosing the Alexander-module generator ``x_0``, following its
deck-transformation orbit ``x_j = t^j x_0``, and recording

``a_j = q * chi(x_j)  (mod q)``.

This is the orbit appearing in Proposition 3.3 of
Conway--Kim--Politarczyk and, after changing its letter from ``a`` to ``b``,
in Yanagida's formulas for twisted Blanchfield pairings.  The same entries
also determine the phase shifts ``a_j/q`` in the divisible-winding branch of
the metabelian satellite formula.

The public homology and character classes use a Smith-normal-form basis,
whereas the orbit is most naturally described in the companion basis of the
Alexander polynomial.  This module performs that basis conversion once and
returns an immutable record.  Keeping it separate prevents the twisted
Alexander and twisted-signature implementations from silently developing
different orbit conventions.

The second public factory, :func:`torus_pattern_phase_orbit`, uses the same
companion and Smith bases for an arbitrary cyclic-cover degree.  It evaluates
a character on the class of ``mu^(-p) eta`` for the standard ``(p,q)`` cable
and on its deck translates.  These exact ``Q/Z`` values are the phase
arguments in BCP-II, Theorem 4.19, including its nondivisible-winding branch.
"""

from dataclasses import dataclass

from sage.all import Integer, QQ, ZZ, gcd, matrix

from gaknot.utils.utility import alexander_polynomial_torus_knot
from gaknot.utils.utility import mod_one


def _validated_torus_parameter(value, name):
    """Return an integer torus parameter and reject booleans and small values."""
    if isinstance(value, bool) or not isinstance(value, (int, Integer)):
        raise TypeError(f"{name} must be an integer.")
    value = Integer(value)
    if value <= 1:
        raise ValueError(f"{name} must be greater than one.")
    return value


def _validated_character_value(value, q, name):
    r"""Normalize one exact ``q``-torsion character value to ``[0,1)``."""
    # Accept Python and Sage exact numbers, but deliberately reject floating
    # point input.  Turning a binary float into a large exact rational would
    # obscure whether a purported root of unity is really exact.
    if isinstance(value, (bool, float, complex, str)) or value is None:
        raise TypeError(f"{name} must be an exact rational number.")
    try:
        value = QQ(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an exact rational number.") from None

    if not (q * value).is_integer():
        raise ValueError(
            f"{name} must define a q-torsion element of Q/Z."
        )
    return value - value.floor()


def _validated_cover_degree(value):
    """Return an exact cyclic-cover degree greater than one."""

    if isinstance(value, bool) or not isinstance(value, (int, Integer)):
        raise TypeError("cover_degree must be an integer.")
    value = Integer(value)
    if value <= 1:
        raise ValueError("cover_degree must be greater than one.")
    return value


def _validated_orbit_length(value, cover_degree):
    """Return a positive initial-orbit length no larger than the cover."""

    if isinstance(value, bool) or not isinstance(value, (int, Integer)):
        raise TypeError("orbit_length must be an integer.")
    value = Integer(value)
    if value <= 0 or value > cover_degree:
        raise ValueError(
            "orbit_length must lie between one and cover_degree."
        )
    return value


def _validated_smith_character_value(value, modulus, name):
    r"""Validate a character value for one public Smith coordinate.

    Positive ``modulus`` represents a cyclic summand ``Z/modulus``.  A zero
    modulus represents a free summand and must receive zero because
    :class:`Character` models only a character on the torsion subgroup.
    """

    if isinstance(value, (bool, float, complex, str)) or value is None:
        raise TypeError(f"{name} must be an exact rational number.")
    try:
        value = QQ(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an exact rational number.") from None

    if modulus == 0:
        if value != 0:
            raise ValueError(
                f"{name} must vanish on a torsion-free Smith summand."
            )
    elif not (modulus * value).is_integer():
        raise ValueError(
            f"{name} is incompatible with the Smith factor {modulus}."
        )
    return QQ(mod_one(value))


def _validated_phase_argument(value, name):
    """Return an exact representative in ``[0,1)`` for a value in ``Q/Z``."""

    if isinstance(value, (bool, float, complex, str)) or value is None:
        raise TypeError(f"{name} must be an exact rational number.")
    try:
        value = QQ(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an exact rational number.") from None
    return QQ(mod_one(value))


@dataclass(frozen=True)
class TorusCharacterOrbit:
    r"""A character in both Smith-generator and deck-orbit coordinates.

    Args:
        p: Cover degree and first torus-knot parameter.
        q: Order of the character roots and second torus-knot parameter.
        generator_values: Images in ``Q/Z`` of the ``p-1`` Smith generators,
            in the order used by :class:`BranchedCoverHomology`.
        a_values: Integer representatives of ``q*chi(x_j)`` for
            ``j=0,...,p-1``.

    ``a_values`` are normalized to ``0,...,q-1``.  They necessarily sum to
    zero modulo ``q``; this is both the consistency relation for the deck
    orbit and the determinant condition required by Yanagida's representation.
    """

    p: object
    q: object
    generator_values: tuple
    a_values: tuple

    def __post_init__(self):
        p = _validated_torus_parameter(self.p, "p")
        q = _validated_torus_parameter(self.q, "q")
        if gcd(p, q) != 1:
            raise ValueError("p and q must be relatively prime.")

        if not isinstance(self.generator_values, (tuple, list)):
            raise TypeError("generator_values must be a tuple or list.")
        if len(self.generator_values) != p - 1:
            raise ValueError(
                f"generator_values must contain exactly p-1={int(p - 1)} entries."
            )
        generator_values = tuple(
            _validated_character_value(value, q, f"generator_values[{index}]")
            for index, value in enumerate(self.generator_values)
        )

        if not isinstance(self.a_values, (tuple, list)):
            raise TypeError("a_values must be a tuple or list.")
        if len(self.a_values) != p:
            raise ValueError(f"a_values must contain exactly p={int(p)} entries.")

        normalized_a = []
        for index, value in enumerate(self.a_values):
            if isinstance(value, bool) or not isinstance(value, (int, Integer)):
                raise TypeError(f"a_values[{index}] must be an integer.")
            normalized_a.append(Integer(value) % q)
        normalized_a = tuple(normalized_a)

        if sum(normalized_a, Integer(0)) % q != 0:
            raise ValueError("The orbit entries must sum to zero modulo q.")

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "generator_values", generator_values)
        object.__setattr__(self, "a_values", normalized_a)

    @property
    def phase_arguments(self):
        r"""Return the exact arguments ``a_j/q`` of the satellite phases."""
        return tuple(QQ(value) / self.q for value in self.a_values)


@dataclass(frozen=True)
class TorusPatternPhaseOrbit:
    r"""Deck orbit of the distinguished class for a standard torus cable.

    Regard ``T(p,q)`` as the pattern of a ``(p,q)`` cable and let ``eta`` be
    the core of the complementary solid torus.  Its winding is ``p``.  The
    commutator element ``mu^(-p) eta`` determines a class ``x_0`` in the
    Alexander module, and its deck translates are ``x_j=t^j x_0``.

    ``smith_coordinates[j]`` gives ``x_j`` in precisely the one-copy Smith
    order used by :class:`BranchedCoverHomology`.  Evaluating
    ``generator_values`` on that tuple produces ``phase_arguments[j]`` in
    ``Q/Z``.  These are the arguments of the roots of unity occurring in the
    companion terms of BCP-II, Theorem 4.19.

    The class is intentionally more general than :class:`TorusCharacterOrbit`:
    the cover degree need not equal ``p``, the Smith factors need not all be
    ``q``, and only an initial part of the full deck orbit may be requested.
    """

    p: object
    q: object
    cover_degree: object
    orbit_length: object
    smith_factors: tuple
    generator_values: tuple
    smith_coordinates: tuple
    phase_arguments: tuple

    def __post_init__(self):
        p = _validated_torus_parameter(self.p, "p")
        q = _validated_torus_parameter(self.q, "q")
        if gcd(p, q) != 1:
            raise ValueError("p and q must be relatively prime.")
        cover_degree = _validated_cover_degree(self.cover_degree)
        orbit_length = _validated_orbit_length(
            self.orbit_length,
            cover_degree,
        )

        if not isinstance(self.smith_factors, (tuple, list)):
            raise TypeError("smith_factors must be a tuple or list.")
        smith_factors = []
        for index, factor in enumerate(self.smith_factors):
            if isinstance(factor, bool) or not isinstance(
                factor,
                (int, Integer),
            ):
                raise TypeError(f"smith_factors[{index}] must be an integer.")
            factor = Integer(factor)
            if factor < 0 or factor == 1:
                raise ValueError(
                    "Smith factors must be zero or greater than one."
                )
            smith_factors.append(factor)
        smith_factors = tuple(smith_factors)

        if not isinstance(self.generator_values, (tuple, list)):
            raise TypeError("generator_values must be a tuple or list.")
        if len(self.generator_values) != len(smith_factors):
            raise ValueError(
                "generator_values and smith_factors must have equal length."
            )
        generator_values = tuple(
            _validated_smith_character_value(
                value,
                smith_factors[index],
                f"generator_values[{index}]",
            )
            for index, value in enumerate(self.generator_values)
        )

        if not isinstance(self.smith_coordinates, (tuple, list)):
            raise TypeError("smith_coordinates must be a tuple or list.")
        if len(self.smith_coordinates) != orbit_length:
            raise ValueError(
                "smith_coordinates must contain orbit_length vectors."
            )
        normalized_coordinates = []
        for deck_power, coordinate_vector in enumerate(
            self.smith_coordinates
        ):
            if not isinstance(coordinate_vector, (tuple, list)):
                raise TypeError(
                    f"smith_coordinates[{deck_power}] must be a tuple or list."
                )
            if len(coordinate_vector) != len(smith_factors):
                raise ValueError(
                    "Every Smith-coordinate vector must have one entry per "
                    "Smith factor."
                )
            normalized_vector = []
            for coordinate in coordinate_vector:
                if isinstance(coordinate, bool) or not isinstance(
                    coordinate,
                    (int, Integer),
                ):
                    raise TypeError("Smith coordinates must be integers.")
                normalized_vector.append(Integer(coordinate))
            normalized_coordinates.append(tuple(normalized_vector))
        normalized_coordinates = tuple(normalized_coordinates)

        if not isinstance(self.phase_arguments, (tuple, list)):
            raise TypeError("phase_arguments must be a tuple or list.")
        if len(self.phase_arguments) != orbit_length:
            raise ValueError(
                "phase_arguments must contain orbit_length entries."
            )
        phase_arguments = tuple(
            _validated_phase_argument(
                argument,
                f"phase_arguments[{index}]",
            )
            for index, argument in enumerate(self.phase_arguments)
        )

        # The record is public and constructible, so verify rather than merely
        # trust that its displayed phases are evaluations of its coordinates.
        expected_arguments = tuple(
            QQ(mod_one(sum(
                (
                    value * coordinate
                    for value, coordinate in zip(
                        generator_values,
                        coordinate_vector,
                    )
                ),
                QQ(0),
            )))
            for coordinate_vector in normalized_coordinates
        )
        if phase_arguments != expected_arguments:
            raise ValueError(
                "phase_arguments do not evaluate the supplied Smith "
                "coordinates."
            )

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "cover_degree", cover_degree)
        object.__setattr__(self, "orbit_length", orbit_length)
        object.__setattr__(self, "smith_factors", smith_factors)
        object.__setattr__(self, "generator_values", generator_values)
        object.__setattr__(
            self,
            "smith_coordinates",
            normalized_coordinates,
        )
        object.__setattr__(self, "phase_arguments", phase_arguments)

    @property
    def deck_powers(self):
        """Return the exponents ``0,...,orbit_length-1`` of the deck action."""

        return tuple(Integer(index) for index in range(int(self.orbit_length)))

    @property
    def distinguished_element_coordinates(self):
        r"""Return the Smith coordinates of ``q_Q(mu_Q^(-p) eta)``."""

        return self.smith_coordinates[0]


def torus_character_orbit(p, q, generator_values):
    r"""Convert Smith-basis character values to the deck orbit for ``T(p,q)``.

    The Alexander polynomial ``Delta`` is monic.  Let ``C`` be its companion
    matrix and let ``D,U,V`` satisfy

    ``U * (C^p-I) * V = D``.

    Then ``C^p-I`` presents the homology of the ``p``-fold branched cover and
    ``U`` converts companion-basis vectors to the Smith basis.  Starting with
    the first companion vector ``x_0``, this function evaluates the supplied
    character on ``x_j=C^j*x_0`` and returns the resulting ``p``-term orbit.

    Args:
        p: The cover degree and first parameter of ``T(p,q)``.
        q: The second torus-knot parameter and orbit modulus.
        generator_values: Images in ``Q/Z`` of the nontrivial Smith
            generators of the cover homology.

    Returns:
        A :class:`TorusCharacterOrbit` containing normalized Smith values,
        integer orbit entries, and exact phase arguments.

    Raises:
        TypeError: If a parameter or character value is not exact data of the
            required type.
        ValueError: If the torus parameters or character coordinates are
            incompatible with ``H_1(Sigma_p(T(p,q)))``.
        ArithmeticError: If the computed Smith form or orbit violates the
            known torus-cover structure.  Such an error signals an internal
            convention or algebra failure rather than invalid user input.
    """
    p = _validated_torus_parameter(p, "p")
    q = _validated_torus_parameter(q, "q")
    if gcd(p, q) != 1:
        raise ValueError("p and q must be relatively prime.")
    if not isinstance(generator_values, (tuple, list)):
        raise TypeError("generator_values must be a tuple or list.")

    # Validate the public coordinates before performing any matrix work.  The
    # p-fold cover has exactly p-1 Smith generators, each of order q.
    if len(generator_values) != p - 1:
        raise ValueError(
            f"generator_values must contain exactly p-1={int(p - 1)} entries."
        )
    normalized_values = tuple(
        _validated_character_value(value, q, f"generator_values[{index}]")
        for index, value in enumerate(generator_values)
    )

    # In the power basis of ZZ[t]/(Delta), multiplication by t is represented
    # by this companion matrix.  Its final column is the monic relation
    # t^d = -sum_{i<d} Delta_i*t^i.
    delta = alexander_polynomial_torus_knot(p, q)
    degree = delta.degree()
    coefficients = delta.list()
    companion = matrix(ZZ, degree, degree)
    for row in range(degree - 1):
        companion[row + 1, row] = 1
    for row in range(degree):
        companion[row, degree - 1] = -coefficients[row]

    # The cover relation is t^p=1.  Smith form diagonalizes its presentation,
    # while the left transformation U expresses an Alexander-module vector in
    # the same generator order used by BranchedCoverHomology and Character.
    presentation = companion ** p - matrix.identity(ZZ, degree)
    diagonal, left_change, _ = presentation.smith_form()
    nontrivial_indices = [
        index
        for index, factor in enumerate(diagonal.diagonal())
        if factor != 1
    ]
    nontrivial_factors = tuple(
        diagonal[index, index] for index in nontrivial_indices
    )

    # This identity is special to the p-fold cover of T(p,q).  Checking it here
    # protects the coordinate conversion: if a later convention change alters
    # the Smith basis, silently pairing values with the wrong generators would
    # be much worse than stopping with a diagnostic error.
    expected_factors = tuple(q for _ in range(p - 1))
    if nontrivial_factors != expected_factors:
        raise ArithmeticError(
            "Unexpected Smith form for the p-fold cover of T(p,q)."
        )

    orbit_values = []
    orbit_vector = matrix(ZZ, degree, 1)
    orbit_vector[0, 0] = 1

    for _ in range(p):
        smith_coordinates = left_change * orbit_vector
        character_value = sum(
            (
                normalized_values[position]
                * smith_coordinates[smith_index, 0]
            )
            for position, smith_index in enumerate(nontrivial_indices)
        )

        # Multiplication by q converts the Q/Z value to its exponent in Z/qZ.
        # Integrality already follows from q-torsion validation, but retaining
        # the explicit check documents the exact bridge between the two models.
        scaled_value = q * character_value
        if not scaled_value.is_integer():
            raise ArithmeticError("A computed orbit value is not q-torsion.")
        orbit_values.append(Integer(scaled_value) % q)

        # Advance once under the deck transformation before the next
        # evaluation.  The order of this update preserves the historical
        # a_0,...,a_{p-1} convention of twisted_alexander_torus_knot.
        orbit_vector = companion * orbit_vector

    # Detect a convention failure before entering the public dataclass, whose
    # ValueError is reserved for callers manually constructing invalid data.
    if sum(orbit_values, Integer(0)) % q != 0:
        raise ArithmeticError("The computed deck orbit does not sum to zero.")

    result = TorusCharacterOrbit(
        p=p,
        q=q,
        generator_values=normalized_values,
        a_values=tuple(orbit_values),
    )
    return result


def torus_pattern_phase_orbit(
    p,
    q,
    cover_degree,
    generator_values,
    orbit_length=None,
):
    r"""Evaluate a character on the standard cable class and its translates.

    Let ``T(p,q)`` be the pattern in the standard ``(p,q)``-cabling
    construction.  With ``eta`` the infection curve and ``mu`` a pattern
    meridian, the class used by BCP-II, Theorem 4.19 is

    ``x_0 = q_Q(mu^(-p) eta)``.

    The torus-knot Alexander module is represented in the power basis of its
    Alexander polynomial, where ``x_0`` is the first basis vector and the deck
    transformation is the companion matrix ``C``.  Thus ``t^j x_0`` is
    represented by ``C^j e_0``.  The left Smith transformation for
    ``C^cover_degree-I`` converts this vector to the public Smith coordinates
    used by :class:`BranchedCoverHomology` and :class:`Character`.

    Args:
        p: First torus parameter and cable winding number.
        q: Second torus parameter, relatively prime to ``p``.
        cover_degree: Degree of the satellite branched cover.
        generator_values: Character values on one copy of the pattern layer,
            in the sorted Smith order exposed by ``BranchedCoverHomology``.
        orbit_length: Number of consecutive translates beginning with
            ``x_0``.  The default returns the full ``cover_degree``-term orbit;
            Theorem 4.19's nondivisible branch requests only
            ``gcd(cover_degree,p)`` terms.

    Returns:
        An immutable :class:`TorusPatternPhaseOrbit` containing the Smith
        factors, the coordinate vector of every requested translate, and its
        exact character value in ``Q/Z``.

    This function does not guess an infection curve for an arbitrary
    satellite.  It implements the distinguished curve in the standard torus
    pattern used by GA-knot cabling descriptions.
    """

    p = _validated_torus_parameter(p, "p")
    q = _validated_torus_parameter(q, "q")
    if gcd(p, q) != 1:
        raise ValueError("p and q must be relatively prime.")
    cover_degree = _validated_cover_degree(cover_degree)
    if orbit_length is None:
        orbit_length = cover_degree
    orbit_length = _validated_orbit_length(orbit_length, cover_degree)
    if not isinstance(generator_values, (tuple, list)):
        raise TypeError("generator_values must be a tuple or list.")

    # This is the same companion presentation used in the branched-homology
    # implementation.  Sharing it is essential: the phase evaluation must not
    # quietly choose a different Alexander-module generator or Smith basis.
    delta = alexander_polynomial_torus_knot(p, q)
    degree = delta.degree()
    coefficients = delta.list()
    companion = matrix(ZZ, degree, degree)
    for row in range(degree - 1):
        companion[row + 1, row] = 1
    for row in range(degree):
        companion[row, degree - 1] = -coefficients[row]

    presentation = (
        companion ** cover_degree - matrix.identity(ZZ, degree)
    )
    diagonal, left_change, _ = presentation.smith_form()

    # BranchedCoverHomology removes unit factors and sorts the remaining
    # factors inside a layer.  Smith form already has a canonical order in the
    # finite cases most often used here, but free factors are diagonal zeros
    # and the explicit sorting moves them to the front.  Carry the original
    # row index along with each factor so character coordinates and the left
    # Smith transformation remain aligned even in that case.
    factor_rows = [
        (diagonal[index, index], index)
        for index in range(degree)
        if diagonal[index, index] != 1
    ]
    factor_rows.sort(key=lambda entry: entry[0])
    smith_factors = tuple(factor for factor, _ in factor_rows)
    smith_row_indices = tuple(index for _, index in factor_rows)

    if len(generator_values) != len(smith_factors):
        raise ValueError(
            "generator_values must contain exactly one entry per nontrivial "
            f"Smith factor ({len(smith_factors)} required)."
        )
    normalized_values = tuple(
        _validated_smith_character_value(
            value,
            smith_factors[index],
            f"generator_values[{index}]",
        )
        for index, value in enumerate(generator_values)
    )

    coordinate_vectors = []
    phase_arguments = []
    orbit_vector = matrix(ZZ, degree, 1)
    orbit_vector[0, 0] = 1

    for _ in range(int(orbit_length)):
        all_smith_coordinates = left_change * orbit_vector
        public_coordinates = tuple(
            Integer(all_smith_coordinates[row_index, 0])
            for row_index in smith_row_indices
        )
        coordinate_vectors.append(public_coordinates)
        phase_arguments.append(QQ(mod_one(sum(
            (
                value * coordinate
                for value, coordinate in zip(
                    normalized_values,
                    public_coordinates,
                )
            ),
            QQ(0),
        ))))
        orbit_vector = companion * orbit_vector

    return TorusPatternPhaseOrbit(
        p=p,
        q=q,
        cover_degree=cover_degree,
        orbit_length=orbit_length,
        smith_factors=smith_factors,
        generator_values=normalized_values,
        smith_coordinates=tuple(coordinate_vectors),
        phase_arguments=tuple(phase_arguments),
    )
