#!/usr/bin/env sage -python

r"""Gilmer genus obstructions for the supported Casson--Gordon family.

Theorem 2.9 of Marchwicka--Politarczyk, following Gilmer and
Florens--Gilmer, says that if a knot ``K`` bounds a locally flat surface of
genus ``g`` in the four-ball, its double-cover linking form splits as
``beta_1 + beta_2`` so that ``beta_1`` has a presentation of rank ``2*g`` and
``beta_2`` is metabolic.  On a metabolizer of ``beta_2`` one must have

.. math::

    |sigma(K, chi_x) + sigma_K| <= eta(K, chi_x) + 4g + 1.

For the GA-knots handled by :mod:`gaknot.invariants.casson_gordon`, every
connected-sum component contributes ``Z/qZ`` with prime ``q`` to the first
homology of the double branched cover.  Its linking form is diagonal, with
coefficient ``-sign/q`` in the distinguished generator.  This makes the
paper's computer-assisted argument both exact and reusable:

* split the form into prime-primary vector spaces;
* enumerate one canonical representative of every nonzero isotropic line;
* for each representative find a nonzero scalar multiple violating Gilmer's
  inequality;
* use a primary part requiring more than ``2*g`` generators to force a
  nonzero element in the hypothetical metabolizer.

The result is a *sufficient obstruction*.  Failure to find the required
violations is reported as ``inconclusive`` and is not evidence that a genus
``g`` surface exists.  Structured result objects retain the first unresolved
isotropic line or a sample successful witness, along with exact counts and
invariant values.
"""

from dataclasses import dataclass
from itertools import product
from math import gcd as integer_gcd

from sage.all import GF, Integer, QQ, is_prime, matrix

from gaknot.core.gaknot import GeneralizedAlgebraicKnot
from gaknot.invariants.casson_gordon import casson_gordon_invariant


def _is_integer(value):
    """Return whether ``value`` is an integer but not a Python boolean."""
    return not isinstance(value, bool) and isinstance(value, (int, Integer))


class PrimeDiagonalLinkingForm:
    r"""A diagonal linking form on a sum of prime-order cyclic groups.

    ``orders[i]`` is the prime order of coordinate ``i`` and
    ``coefficients[i]`` is the numerator of its self-pairing.  Thus

    .. math::

        lambda(x,y) = sum_i coefficients[i] x_i y_i / orders[i]
                       \quad\text{in } Q/Z.

    Coefficients must be units modulo their corresponding prime.  The class is
    intentionally restricted to this domain: it is exactly the linking-form
    structure supplied by the Phase 1 ``(2,q)``-cable formulas, and it permits
    metabolizer checks by finite-field linear algebra rather than subgroup
    enumeration.
    """

    def __init__(self, orders, coefficients):
        """Validate and store an immutable diagonal presentation."""
        if not isinstance(orders, (list, tuple)):
            raise TypeError("Linking-form orders must be a list or tuple.")
        if not isinstance(coefficients, (list, tuple)):
            raise TypeError(
                "Linking-form coefficients must be a list or tuple."
            )
        if len(orders) == 0:
            raise ValueError("A linking form must contain at least one coordinate.")
        if len(orders) != len(coefficients):
            raise ValueError(
                "Linking-form orders and coefficients must have equal length."
            )

        normalized_orders = []
        normalized_coefficients = []

        for index, (order, coefficient) in enumerate(zip(orders, coefficients)):
            if not _is_integer(order):
                raise TypeError(f"Order at index {index} must be an integer.")
            if not is_prime(order):
                raise ValueError(
                    f"Order at index {index} must be prime; got {order}."
                )
            if not _is_integer(coefficient):
                raise TypeError(
                    f"Coefficient at index {index} must be an integer."
                )
            if integer_gcd(int(coefficient), int(order)) != 1:
                raise ValueError(
                    f"Coefficient {coefficient} at index {index} is not a "
                    f"unit modulo {order}."
                )

            normalized_orders.append(Integer(order))
            # Retain the signed representative because +/-1 records the knot
            # orientation transparently, even though arithmetic is modulo q.
            normalized_coefficients.append(Integer(coefficient))

        self._orders = tuple(normalized_orders)
        self._coefficients = tuple(normalized_coefficients)

    @classmethod
    def from_knot(cls, knot):
        r"""Construct the double-cover linking form of a supported GA-knot.

        A positive ``(2,q)``-cable has distinguished self-pairing ``-1/q``;
        taking the concordance inverse reverses that sign.  Calling the public
        Casson--Gordon function with the zero character validates every
        component and supplies its prime ``q`` without duplicating Phase 1's
        cabling-domain rules.
        """
        if not isinstance(knot, GeneralizedAlgebraicKnot):
            raise TypeError(
                "Expected a GeneralizedAlgebraicKnot object, "
                f"got {type(knot)}."
            )

        zero_result = casson_gordon_invariant(knot, [0] * len(knot))
        orders = [summand.q for summand in zero_result.summands]
        coefficients = [-sign for sign, _ in knot.description]
        return cls(orders, coefficients)

    @property
    def orders(self):
        """Return the immutable tuple of cyclic coordinate orders."""
        return self._orders

    @property
    def coefficients(self):
        """Return the immutable tuple of diagonal numerators."""
        return self._coefficients

    @property
    def rank(self):
        """Return the number of diagonal cyclic coordinates."""
        return len(self._orders)

    @property
    def primary_primes(self):
        """Return the occurring primes in increasing order."""
        return tuple(sorted(set(self._orders)))

    def __repr__(self):
        """Return the complete diagonal data needed to recreate the form."""
        return (
            "PrimeDiagonalLinkingForm("
            f"orders={self._orders!r}, coefficients={self._coefficients!r})"
        )

    def primary_indices(self, prime):
        """Return the coordinate indices belonging to one primary part."""
        if not _is_integer(prime):
            raise TypeError("The primary order must be an integer.")
        prime = Integer(prime)
        return tuple(
            index for index, order in enumerate(self._orders)
            if order == prime
        )

    def _normalize_element(self, element):
        """Validate and reduce one coordinate vector modulo its orders."""
        if not isinstance(element, (list, tuple)):
            raise TypeError("A linking-form element must be a list or tuple.")
        if len(element) != self.rank:
            raise ValueError(
                f"Expected {self.rank} element coordinates, "
                f"but received {len(element)}."
            )

        normalized = []
        for index, (coordinate, order) in enumerate(zip(element, self._orders)):
            if not _is_integer(coordinate):
                raise TypeError(
                    f"Element coordinate at index {index} must be an integer."
                )
            normalized.append(Integer(coordinate) % order)
        return tuple(normalized)

    def pairing(self, left, right):
        r"""Return ``lambda(left,right)`` in the representative interval ``[0,1)``."""
        left = self._normalize_element(left)
        right = self._normalize_element(right)

        value = sum(
            (
                QQ(coefficient * x_value * y_value) / order
                for coefficient, order, x_value, y_value in zip(
                    self._coefficients,
                    self._orders,
                    left,
                    right,
                )
            ),
            QQ(0),
        )
        return value - value.floor()

    def is_isotropic(self, element):
        """Return whether ``lambda(element,element)`` is zero in ``Q/Z``."""
        return self.pairing(element, element) == 0

    def is_metabolizer(self, generators):
        r"""Return whether the supplied vectors generate a metabolizer.

        On the ``p``-primary part, a metabolizer is a totally isotropic
        subspace of dimension half the ambient dimension.  Generator vectors
        may simultaneously have coordinates in several primary parts; their
        projections are checked independently, exactly as in the primary
        decomposition of a finite linking form.
        """
        if not isinstance(generators, (list, tuple)):
            raise TypeError("Metabolizer generators must be a list or tuple.")

        normalized_generators = [
            self._normalize_element(generator) for generator in generators
        ]

        for prime in self.primary_primes:
            indices = self.primary_indices(prime)
            dimension = len(indices)
            field = GF(prime)
            rows = [
                [generator[index] for index in indices]
                for generator in normalized_generators
            ]
            generator_matrix = matrix(field, rows) if rows else matrix(
                field,
                0,
                dimension,
            )

            # A half-dimensional subspace is necessary for M=M^perp in a
            # nonsingular form.  This also rejects odd-dimensional blocks.
            if 2 * generator_matrix.rank() != dimension:
                return False

            diagonal = matrix.diagonal(
                field,
                [self._coefficients[index] for index in indices],
            )
            gram_matrix = (
                generator_matrix
                * diagonal
                * generator_matrix.transpose()
            )
            if not gram_matrix.is_zero():
                return False

        return True

    def projective_isotropic_elements(self, prime):
        r"""Yield one canonical representative of each isotropic line.

        A nonzero vector over ``F_p`` has a unique scalar multiple whose first
        nonzero coordinate is one.  Enumerating only those representatives
        removes the ``p-1`` redundant nonzero multiples while preserving the
        property tested by the genus obstruction.

        Returned vectors have the full coordinate length of this form and are
        zero outside the selected primary part.
        """
        indices = self.primary_indices(prime)
        if not indices:
            raise ValueError(f"Prime {prime} does not occur in this linking form.")

        prime = Integer(prime)
        local_dimension = len(indices)
        prime_as_int = int(prime)

        for pivot in range(local_dimension):
            tail_length = local_dimension - pivot - 1
            for tail in product(range(prime_as_int), repeat=tail_length):
                local_vector = (0,) * pivot + (1,) + tail
                local_self_pairing = sum(
                    self._coefficients[global_index]
                    * local_vector[local_index]
                    * local_vector[local_index]
                    for local_index, global_index in enumerate(indices)
                )
                if local_self_pairing % prime != 0:
                    continue

                full_vector = [Integer(0)] * self.rank
                for local_index, global_index in enumerate(indices):
                    full_vector[global_index] = Integer(
                        local_vector[local_index]
                    )
                yield tuple(full_vector)


@dataclass(frozen=True)
class GilmerViolationWitness:
    """One multiple of an isotropic element violating Gilmer's inequality."""

    prime: object
    isotropic_element: tuple
    multiple: object
    character_parameters: tuple
    sigma: object
    eta: object
    classical_signature: object
    left_hand_side: object
    bound: object


@dataclass(frozen=True)
class PrimaryGenusCheck:
    """Search summary for one prime-primary part of the linking form."""

    prime: object
    component_indices: tuple
    generator_count: int
    eligible: bool
    projective_vectors_in_search_space: int
    isotropic_lines_examined: int
    violating_lines: int
    certified: bool
    sample_witness: object = None
    unresolved_isotropic_element: object = None


@dataclass(frozen=True)
class GenusObstructionResult:
    r"""Auditable result of the supported Gilmer genus search.

    ``certified=True`` proves that the topological four-genus is greater than
    ``tested_genus``.  ``certified=False`` means only that this sufficient
    search did not prove that lower bound.
    """

    tested_genus: object
    classical_signature: object
    linking_form: PrimeDiagonalLinkingForm
    primary_checks: tuple
    certified: bool

    @property
    def lower_bound(self):
        """Return the certified integral lower bound, or ``None``."""
        if not self.certified:
            return None
        return self.tested_genus + 1

    @property
    def successful_primes(self):
        """Return primary orders that individually certify the obstruction."""
        return tuple(
            check.prime for check in self.primary_checks if check.certified
        )


def _scaled_component_signature_tables(knot, prime, component_indices):
    """Precompute ``prime*sigma`` as Python integers for the inner search."""
    tables = {}
    prime_as_int = int(prime)

    for component_index in component_indices:
        component = knot[component_index]
        table = []
        for parameter in range(prime_as_int):
            sigma = component.casson_gordon(parameter).sigma
            scaled_sigma = sigma * prime
            if not scaled_sigma.is_integer():
                raise ArithmeticError(
                    "Casson--Gordon signature did not have the expected "
                    f"denominator {prime}."
                )
            table.append(int(scaled_sigma))
        tables[component_index] = tuple(table)

    return tables


def _character_parameters_for_multiple(
    linking_form,
    isotropic_element,
    prime,
    multiple,
):
    r"""Convert a homology element into parameters of ``chi_x``.

    In the distinguished diagonal basis, evaluation on generator ``i`` is
    ``coefficient_i*x_i/q``.  Components outside the selected primary part
    remain trivial.
    """
    parameters = [Integer(0)] * linking_form.rank
    for index in linking_form.primary_indices(prime):
        parameters[index] = Integer(
            linking_form.coefficients[index]
            * multiple
            * isotropic_element[index]
        ) % prime
    return tuple(parameters)


def _check_primary_part(
    knot,
    linking_form,
    prime,
    genus,
    classical_signature,
):
    """Run the projective isotropic-line search for one prime."""
    component_indices = linking_form.primary_indices(prime)
    generator_count = len(component_indices)
    eligible = generator_count > 2 * genus

    if not eligible:
        return PrimaryGenusCheck(
            prime=prime,
            component_indices=component_indices,
            generator_count=generator_count,
            eligible=False,
            projective_vectors_in_search_space=0,
            isotropic_lines_examined=0,
            violating_lines=0,
            certified=False,
        )

    signature_tables = _scaled_component_signature_tables(
        knot,
        prime,
        component_indices,
    )
    isotropic_lines_examined = 0
    violating_lines = 0
    sample_witness = None
    prime_as_int = int(prime)

    # The generator yields isotropic canonical vectors only.  Count the number
    # of all projective vectors separately from the closed formula so the
    # summary makes the optimization visible without enumerating nonisotropic
    # vectors a second time.
    projective_vectors_in_search_space = sum(
        prime_as_int ** exponent for exponent in range(generator_count)
    )

    for isotropic_element in linking_form.projective_isotropic_elements(prime):
        isotropic_lines_examined += 1
        nonzero_count = sum(
            1 for index in component_indices
            if isotropic_element[index] != 0
        )
        # Individual prime-order cable nullities vanish.  Connecting r
        # nontrivial restrictions adds r-1, as implemented in Phase 1.
        eta = Integer(max(nonzero_count - 1, 0))
        bound = eta + 4 * genus + 1
        scaled_bound = prime_as_int * int(bound)
        line_witness = None

        # Multipliers k and p-k negate every character parameter.  The
        # Casson--Gordon formula is unchanged by this simultaneous conjugation,
        # so one representative from each pair suffices.
        for multiple in range(1, prime_as_int // 2 + 1):
            character_parameters = _character_parameters_for_multiple(
                linking_form,
                isotropic_element,
                prime,
                multiple,
            )
            scaled_sigma = sum(
                signature_tables[index][int(character_parameters[index])]
                for index in component_indices
            )
            scaled_left_hand_side = abs(
                scaled_sigma + prime_as_int * int(classical_signature)
            )

            if scaled_left_hand_side > scaled_bound:
                sigma = QQ(scaled_sigma) / prime
                left_hand_side = QQ(scaled_left_hand_side) / prime
                line_witness = GilmerViolationWitness(
                    prime=prime,
                    isotropic_element=isotropic_element,
                    multiple=Integer(multiple),
                    character_parameters=character_parameters,
                    sigma=sigma,
                    eta=eta,
                    classical_signature=classical_signature,
                    left_hand_side=left_hand_side,
                    bound=bound,
                )
                break

        if line_witness is None:
            return PrimaryGenusCheck(
                prime=prime,
                component_indices=component_indices,
                generator_count=generator_count,
                eligible=True,
                projective_vectors_in_search_space=(
                    projective_vectors_in_search_space
                ),
                isotropic_lines_examined=isotropic_lines_examined,
                violating_lines=violating_lines,
                certified=False,
                sample_witness=sample_witness,
                unresolved_isotropic_element=isotropic_element,
            )

        violating_lines += 1
        if sample_witness is None:
            sample_witness = line_witness

    # An eligible anisotropic block also certifies the obstruction: a nonzero
    # metabolic primary summand would supply a nonzero isotropic metabolizer
    # element, but none exists.
    return PrimaryGenusCheck(
        prime=prime,
        component_indices=component_indices,
        generator_count=generator_count,
        eligible=True,
        projective_vectors_in_search_space=projective_vectors_in_search_space,
        isotropic_lines_examined=isotropic_lines_examined,
        violating_lines=violating_lines,
        certified=True,
        sample_witness=sample_witness,
    )


def gilmer_genus_obstruction(knot, genus):
    r"""Try to certify ``g_4^top(knot) > genus`` using Gilmer's bound.

    The search is complete for the sufficient primary-isotropic criterion used
    in the Marchwicka--Politarczyk genus-two example.  It is intentionally not
    advertised as a decision procedure for four-genus.

    Args:
        knot: A GA-knot supported by ``casson_gordon_invariant``.
        genus: A nonnegative integer genus whose existence is to be obstructed.

    Returns:
        A ``GenusObstructionResult``.  Its ``certified`` flag is true exactly
        when at least one eligible primary part proves the requested strict
        lower bound.
    """
    if not isinstance(knot, GeneralizedAlgebraicKnot):
        raise TypeError(
            "Expected a GeneralizedAlgebraicKnot object, "
            f"got {type(knot)}."
        )
    if not _is_integer(genus):
        raise TypeError("The tested genus must be an integer.")
    if genus < 0:
        raise ValueError("The tested genus must be nonnegative.")
    genus = Integer(genus)

    linking_form = PrimeDiagonalLinkingForm.from_knot(knot)
    classical_signature = Integer(knot.signature()(QQ(1) / 2))
    primary_checks = tuple(
        _check_primary_part(
            knot,
            linking_form,
            prime,
            genus,
            classical_signature,
        )
        for prime in linking_form.primary_primes
    )
    certified = any(check.certified for check in primary_checks)

    return GenusObstructionResult(
        tested_genus=genus,
        classical_signature=classical_signature,
        linking_form=linking_form,
        primary_checks=primary_checks,
        certified=certified,
    )
