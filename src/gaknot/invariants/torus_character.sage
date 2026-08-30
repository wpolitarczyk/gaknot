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
"""

from dataclasses import dataclass

from sage.all import Integer, QQ, ZZ, gcd, matrix

from gaknot.utils.utility import alexander_polynomial_torus_knot


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
