#!/usr/bin/env sage -python

r"""Computational ingredients from Conway--Kim--Politarczyk.

This module implements the finite calculations in Sections 3 and 5.1 of
Anthony Conway, Min Hoon Kim, and Wojciech Politarczyk,
*Non-slice linear combinations of iterated torus knots*.

For a character on the ``p``-fold branched cover of ``T(p,q)``, Proposition
3.2 conjugates the metabelian representation to the particularly transparent
form

.. math::

    c_1 \longmapsto A_p(t)^q,
    \qquad
    c_2 \longmapsto
        t\,\operatorname{diag}(\zeta_q^{a_0},\ldots,
                               \zeta_q^{a_{p-1}}),

where ``A_p(t)`` is the cyclic shift matrix satisfying
``A_p(t)^p=t I``.  The relation ``c_1^p=c_2^q`` is then visible directly.
The determinants in the Fox-calculus quotient give Proposition 3.3 for the
knot exterior and Corollary 3.4 for zero-framed surgery.

The :class:`CKPTorusKnotData` record retains every one of these intermediate
objects.  This is intentional: a paper-reconstruction notebook should be able
to display and audit the matrices and determinants, not only the final
rational function.

The second part of the module implements the paper's ``s``-levels for a
signed sum of iterated torus knots with common first cabling parameter ``p``.
The level record preserves expanded summands as well as their formal signed
multiplicities.  Consequently a cancellation such as
``T(p,q) # -T(p,q)`` is visible both before and after collection.

This is not an implementation of the complete proof of Theorem 5.1.  In
particular, it does not construct equivariant linking forms, invariant
metabolizers, graph anti-isometries, or Witt classes.  The records here cover
the exact matrix, polynomial, root-support, and cabling-level calculations on
which that later layer will build.
"""

from dataclasses import dataclass, field

from sage.all import (
    CyclotomicField,
    Integer,
    PolynomialRing,
    gcd,
    identity_matrix,
    matrix,
)

from gaknot.invariants.torus_character import (
    TorusCharacterOrbit,
    torus_character_orbit,
)


def _immutable(matrix_value):
    """Mark a Sage matrix immutable before storing it in a frozen record."""
    matrix_value.set_immutable()
    return matrix_value


def _validate_positive_torus_character(knot, character):
    r"""Validate the common domain of the CKP torus-knot formulas.

    The character coordinates are meaningful only in the Smith basis of the
    supplied knot.  Comparing the complete structural descriptions permits an
    independently constructed but equivalent knot object, while rejecting a
    character belonging to a different torus knot with coincidentally similar
    homology.
    """
    from gaknot.core.gaknot import GeneralizedAlgebraicKnot
    from gaknot.invariants.character import Character

    if not isinstance(knot, GeneralizedAlgebraicKnot):
        raise TypeError(
            "knot must be a GeneralizedAlgebraicKnot object."
        )
    if knot.is_negative_torus_knot():
        raise NotImplementedError(
            "Twisted Alexander polynomial for negative torus knots not yet implemented."
        )
    if not knot.is_positive_torus_knot():
        raise ValueError("Knot must be a positive torus knot T(p,q).")
    if not isinstance(character, Character):
        raise TypeError(f"Expected a Character object, got {type(character)}.")

    _, cable_description = knot.description[0]
    p, q = cable_description[0]
    homology = character.homology
    if homology.knot.description != knot.description:
        raise ValueError(
            "Character must be defined on the homology of the supplied knot."
        )
    if homology.cover_degree != p:
        raise ValueError(
            f"Formula requires character on the {p}-fold cover "
            f"(the first torus parameter). Got N={homology.cover_degree}, "
            f"expected p={p}."
        )

    generator_values = character.restrict_to_layer(0, 0)[0]
    return torus_character_orbit(p, q, generator_values)


@dataclass(frozen=True)
class CKPRootMultiplicity:
    r"""Multiplicity of one ``q``-th root in a fixed CKP representative.

    ``exponent`` denotes the root ``zeta_q^exponent``.  Positive
    ``multiplicity`` means a zero, a negative value means a pole, and zero
    records complete cancellation.  Retaining poles is important for trivial
    characters: the displayed twisted-Alexander representatives are rational
    functions and should not silently be treated as polynomials.
    """

    modulus: object
    exponent: object
    multiplicity: object

    def __post_init__(self):
        for name in ("modulus", "exponent", "multiplicity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, Integer)):
                raise TypeError(f"{name} must be an integer.")

        modulus = Integer(self.modulus)
        if modulus <= 1:
            raise ValueError("modulus must be greater than one.")

        object.__setattr__(self, "modulus", modulus)
        object.__setattr__(self, "exponent", Integer(self.exponent) % modulus)
        object.__setattr__(self, "multiplicity", Integer(self.multiplicity))

    @property
    def root_order(self):
        """Return the multiplicative order of ``zeta_q^exponent``."""
        if self.exponent == 0:
            return Integer(1)
        return self.modulus // gcd(self.modulus, self.exponent)

    @property
    def kind(self):
        """Classify the divisor coefficient as ``zero``, ``pole``, or neither."""
        if self.multiplicity > 0:
            return "zero"
        if self.multiplicity < 0:
            return "pole"
        return "cancelled"


@dataclass(frozen=True)
class CKPTorusKnotData:
    r"""Proposition 3.2 matrices and their twisted-Alexander consequences.

    Args:
        orbit: A :class:`TorusCharacterOrbit` encoding the character
            ``chi_a`` on ``H_1(Sigma_p(T(p,q)))``.

    All matrices live in ``Q(zeta_q)(t)`` and are immutable.  The fields
    ``fox_numerator_determinant`` and ``fox_denominator_determinant`` are the
    two determinants in equation (12) of the paper.  Their quotient is the
    fixed exterior representative from Proposition 3.3; multiplying by
    ``(-1)^(p-1)/(t-1)`` gives the zero-surgery representative in Corollary
    3.4.
    """

    orbit: TorusCharacterOrbit
    p: object = field(init=False)
    q: object = field(init=False)
    coefficient_field: object = field(init=False, repr=False)
    polynomial_ring: object = field(init=False, repr=False)
    function_field: object = field(init=False, repr=False)
    t: object = field(init=False, repr=False)
    zeta: object = field(init=False, repr=False)
    A: object = field(init=False, repr=False)
    c1_image: object = field(init=False, repr=False)
    c2_image: object = field(init=False, repr=False)
    fox_numerator_matrix: object = field(init=False, repr=False)
    fox_numerator_determinant: object = field(init=False, repr=False)
    fox_denominator_determinant: object = field(init=False, repr=False)
    exterior_twisted_alexander: object = field(init=False, repr=False)
    zero_surgery_twisted_alexander: object = field(init=False, repr=False)
    exterior_root_multiplicities: tuple = field(init=False)
    zero_surgery_root_multiplicities: tuple = field(init=False)

    def __post_init__(self):
        if not isinstance(self.orbit, TorusCharacterOrbit):
            raise TypeError("orbit must be a TorusCharacterOrbit object.")

        p = self.orbit.p
        q = self.orbit.q
        coefficient_field = CyclotomicField(q)
        zeta = coefficient_field.gen()
        polynomial_ring = PolynomialRing(coefficient_field, "t")
        function_field = polynomial_ring.fraction_field()
        t = function_field.gen()

        # This is A_p(t) from equation (8): it cyclically shifts the standard
        # basis and multiplies the vector wrapping from the final coordinate
        # back to the first by t.  Hence A^p=tI.
        A = matrix(function_field, p, p)
        for row in range(p - 1):
            A[row, row + 1] = 1
        A[p - 1, 0] = t

        # Proposition 3.2 gives these images after a simultaneous conjugation.
        # Conjugation does not change either determinant used by Fox calculus.
        c1_image = A ** q
        c2_image = t * matrix.diagonal(
            function_field,
            [zeta ** coordinate for coordinate in self.orbit.a_values],
        )

        # The torus-knot group has relator c1^p c2^(-q).  Check its image at
        # construction time so a convention error cannot contaminate the
        # subsequent determinant formulas.
        identity = identity_matrix(function_field, p)
        if c1_image ** p != c2_image ** q:
            raise ArithmeticError("The CKP matrices do not satisfy c1^p=c2^q.")
        if A ** p != t * identity:
            raise ArithmeticError("The shift matrix does not satisfy A^p=tI.")

        # Fox differentiation of c1^p c2^(-q) with respect to c1 gives
        # 1+c1+...+c1^(p-1).  Its determinant is equation (14).
        fox_numerator_matrix = identity_matrix(function_field, p)
        power = identity_matrix(function_field, p)
        for _ in range(1, p):
            power *= c1_image
            fox_numerator_matrix += power
        fox_numerator_determinant = fox_numerator_matrix.det()
        expected_numerator = (1 - t ** q) ** (p - 1)
        if fox_numerator_determinant != expected_numerator:
            raise ArithmeticError(
                "The Fox numerator determinant disagrees with equation (14)."
            )

        # Deleting the c2 column in the Fox formula contributes det(c2-I) in
        # the denominator; diagonal form makes equation (13) immediate.
        fox_denominator_determinant = (c2_image - identity).det()
        expected_denominator = function_field.one()
        for coordinate in self.orbit.a_values:
            expected_denominator *= t * zeta ** coordinate - 1
        if fox_denominator_determinant != expected_denominator:
            raise ArithmeticError(
                "The Fox denominator determinant disagrees with equation (13)."
            )

        exterior = fox_numerator_determinant / fox_denominator_determinant
        zero_surgery = (-1) ** (p - 1) * exterior / (t - 1)

        exterior_roots = self._root_divisor(p, q, self.orbit.a_values, False)
        surgery_roots = self._root_divisor(p, q, self.orbit.a_values, True)

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "coefficient_field", coefficient_field)
        object.__setattr__(self, "polynomial_ring", polynomial_ring)
        object.__setattr__(self, "function_field", function_field)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "zeta", zeta)
        object.__setattr__(self, "A", _immutable(A))
        object.__setattr__(self, "c1_image", _immutable(c1_image))
        object.__setattr__(self, "c2_image", _immutable(c2_image))
        object.__setattr__(
            self,
            "fox_numerator_matrix",
            _immutable(fox_numerator_matrix),
        )
        object.__setattr__(
            self,
            "fox_numerator_determinant",
            fox_numerator_determinant,
        )
        object.__setattr__(
            self,
            "fox_denominator_determinant",
            fox_denominator_determinant,
        )
        object.__setattr__(self, "exterior_twisted_alexander", exterior)
        object.__setattr__(
            self,
            "zero_surgery_twisted_alexander",
            zero_surgery,
        )
        object.__setattr__(
            self,
            "exterior_root_multiplicities",
            exterior_roots,
        )
        object.__setattr__(
            self,
            "zero_surgery_root_multiplicities",
            surgery_roots,
        )

    @staticmethod
    def _root_divisor(p, q, a_values, zero_surgery):
        r"""Return the divisor at every ``q``-th root of unity.

        The numerator contributes ``p-1`` at every ``q``-th root.  A factor
        ``t*zeta_q^a-1`` removes one copy at exponent ``-a``, and the surgery
        factor ``t-1`` removes one additional copy at exponent zero.
        """
        result = []
        for exponent in range(q):
            denominator_count = sum(
                Integer(coordinate == (-exponent) % q)
                for coordinate in a_values
            )
            multiplicity = p - 1 - denominator_count
            if zero_surgery and exponent == 0:
                multiplicity -= 1
            result.append(CKPRootMultiplicity(q, exponent, multiplicity))
        return tuple(result)

    @property
    def relation_holds(self):
        """Return the directly verified torus-group relation."""
        return self.c1_image ** self.p == self.c2_image ** self.q

    @property
    def exterior_zero_support(self):
        """Return roots occurring with positive multiplicity in Proposition 3.3."""
        return tuple(
            root
            for root in self.exterior_root_multiplicities
            if root.multiplicity > 0
        )

    @property
    def zero_surgery_zero_support(self):
        """Return roots occurring with positive multiplicity in Corollary 3.4."""
        return tuple(
            root
            for root in self.zero_surgery_root_multiplicities
            if root.multiplicity > 0
        )

    @property
    def exterior_pole_support(self):
        """Return poles of the fixed exterior representative."""
        return tuple(
            root
            for root in self.exterior_root_multiplicities
            if root.multiplicity < 0
        )

    @property
    def zero_surgery_pole_support(self):
        """Return poles of the fixed zero-surgery representative."""
        return tuple(
            root
            for root in self.zero_surgery_root_multiplicities
            if root.multiplicity < 0
        )


def ckp_torus_knot_data(knot, character):
    r"""Build all CKP Section 3 data from a public knot and character."""
    orbit = _validate_positive_torus_character(knot, character)
    return CKPTorusKnotData(orbit)


def zero_surgery_twisted_alexander_torus_knot(knot, character):
    r"""Return the fixed Corollary 3.4 representative for ``M_T(p,q)``."""
    return ckp_torus_knot_data(knot, character).zero_surgery_twisted_alexander


@dataclass(frozen=True)
class CKPLevelTerm:
    r"""One expanded torus-knot summand in an ``s``-level.

    ``source_component`` and ``source_layer`` identify exactly where the term
    came from in the original GA-knot description.  ``root_modulus`` records
    the modulus ``p^(s+1) q`` in the root-separation calculation in the proof
    of Proposition 5.4.
    """

    sign: object
    p: object
    q: object
    source_component: object
    source_layer: object
    root_modulus: object

    def __post_init__(self):
        integer_fields = (
            "sign",
            "p",
            "q",
            "source_component",
            "source_layer",
            "root_modulus",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, Integer)):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, Integer(value))
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1.")
        if self.p <= 1 or self.q <= 1 or gcd(self.p, self.q) != 1:
            raise ValueError("p and q must be coprime integers greater than one.")
        if self.source_component < 0 or self.source_layer < 0:
            raise ValueError("source indices must be nonnegative.")
        if self.root_modulus <= 1:
            raise ValueError("root_modulus must be greater than one.")


@dataclass(frozen=True)
class CKPCableLevel:
    r"""The expanded and collected data for one CKP ``s``-level."""

    s: object
    p: object
    substitution_power: object
    terms: tuple
    signed_multiplicities: tuple

    def __post_init__(self):
        for name in ("s", "p", "substitution_power"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, Integer)):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, Integer(value))
        if self.s < 0:
            raise ValueError("s must be nonnegative.")
        if self.p <= 1:
            raise ValueError("p must be greater than one.")
        if self.substitution_power != self.p ** self.s:
            raise ValueError("substitution_power must equal p^s.")
        if not isinstance(self.terms, tuple) or not all(
            isinstance(term, CKPLevelTerm) for term in self.terms
        ):
            raise TypeError("terms must be a tuple of CKPLevelTerm objects.")
        if not isinstance(self.signed_multiplicities, tuple):
            raise TypeError("signed_multiplicities must be a tuple.")

        if any(term.p != self.p for term in self.terms):
            raise ValueError("Every level term must use the level's common p.")

        # ``signed_multiplicities`` is derived data, but the dataclass remains
        # publicly constructible.  Verify that it is the sorted collection of
        # the expanded terms so the formal-cancellation predicate can never
        # report an answer inconsistent with ``terms``.
        expected_by_q = {}
        for term in self.terms:
            expected_by_q[term.q] = (
                expected_by_q.get(term.q, Integer(0)) + term.sign
            )
        expected_multiplicities = tuple(
            (q, expected_by_q[q]) for q in sorted(expected_by_q)
        )
        if self.signed_multiplicities != expected_multiplicities:
            raise ValueError(
                "signed_multiplicities must be the sorted signed collection "
                "of terms."
            )

    @property
    def is_formally_zero(self):
        """Return whether all torus summands cancel with their inverses."""
        return all(coefficient == 0 for _, coefficient in self.signed_multiplicities)

    @property
    def root_moduli(self):
        """Return the distinct root moduli appearing before cancellation."""
        return tuple(sorted({term.root_modulus for term in self.terms}))


def ckp_cable_levels(knot):
    r"""Return all nonempty CKP ``s``-levels of a GA-knot description.

    Every cabling pair must have the same first parameter ``p``, as in the
    family ``T(p,q_1; ...; p,q_l)`` studied in Section 5.  If a component has
    length at most ``s``, its contribution to level ``s`` is the unknot and is
    omitted.  Levels at and above the maximum cable length are therefore
    implicitly empty and are not returned.
    """
    from gaknot.core.gaknot import GeneralizedAlgebraicKnot

    if not isinstance(knot, GeneralizedAlgebraicKnot):
        raise TypeError("knot must be a GeneralizedAlgebraicKnot object.")

    description = knot.description
    common_p = Integer(description[0][1][0][0])
    for component_index, (_, sequence) in enumerate(description):
        for layer_index, (p, _) in enumerate(sequence):
            if p != common_p:
                raise ValueError(
                    "Every cabling layer must use the same first parameter p; "
                    f"component {component_index}, layer {layer_index} uses {p} "
                    f"instead of {common_p}."
                )

    maximum_length = max(len(sequence) for _, sequence in description)
    levels = []
    for s in range(maximum_length):
        terms = []
        coefficient_by_q = {}
        for component_index, (sign, sequence) in enumerate(description):
            if len(sequence) <= s:
                continue
            source_layer = len(sequence) - 1 - s
            p, q = sequence[source_layer]
            term = CKPLevelTerm(
                sign=sign,
                p=p,
                q=q,
                source_component=component_index,
                source_layer=source_layer,
                root_modulus=Integer(p) ** (s + 1) * Integer(q),
            )
            terms.append(term)
            coefficient_by_q[Integer(q)] = (
                coefficient_by_q.get(Integer(q), Integer(0)) + sign
            )

        signed_multiplicities = tuple(
            (q, coefficient_by_q[q]) for q in sorted(coefficient_by_q)
        )
        levels.append(CKPCableLevel(
            s=s,
            p=common_p,
            substitution_power=common_p ** s,
            terms=tuple(terms),
            signed_multiplicities=signed_multiplicities,
        ))
    return tuple(levels)
