#!/usr/bin/env sage -python

r"""Casson--Gordon signatures for double covers of ``(2,q)``-cables.

This module implements the cabling formula used in Marchwicka--Politarczyk,
*On the slice genus of generalized algebraic knots*, Lemma 2.12.  If ``q`` is
an odd prime, ``K(2,q)`` is the ``(2,q)``-cable of a knot ``K``, and the
distinguished character ``chi_a`` sends the standard generator of
``H_1(Sigma_2(K(2,q))) = Z/qZ`` to ``exp(2*pi*i*a/q)``, then

.. math::

    sigma(K(2,q), chi_a)
      = -q + 2a(q-a)/q + 2 sigma_K(exp(2*pi*i*a/q))

for ``a != 0``.  Both the signature and nullity vanish for the trivial
character.  For a nontrivial character the cabling formula also gives

.. math::

    eta(K(2,q), chi_a) = 2 eta_K(exp(2*pi*i*a/q)).

The present API deliberately accepts the integer ``a`` rather than a
``Character`` object.  ``Character`` coordinates use a Smith basis computed
from a presentation matrix, while the formula above uses a geometrically
distinguished lens-space generator.  Identifying those bases requires linking
form data and belongs to the later genus-obstruction phase.

The supported knots are signed connected sums of iterated torus knots whose
outermost operation is a ``(2,q)``-cable with prime ``q``.  An ordinary
``T(2,q)`` is the case with no companion.  For an ordinary torus knot only,
the symmetric spelling ``T(q,2)`` is also accepted; in a genuine cable the
first coordinate is the winding number and therefore must literally be two.

All signature calculations use exact Sage integers and rationals.  The
returned immutable objects retain the pattern and satellite contributions so
that a genus-bound calculation can explain rather than merely report its
answer.
"""

from dataclasses import dataclass

from sage.all import Integer, QQ, is_prime

from gaknot.core.gaknot import GeneralizedAlgebraicKnot
from gaknot.invariants.LT_signature import LT_signature_iterated_torus_knot


def _is_integer(value):
    """Return whether ``value`` is an accepted exact integer input.

    Booleans need explicit rejection because Python makes ``bool`` a subclass
    of ``int``.  Treating ``True`` as the character parameter one would hide a
    likely input mistake.
    """
    return not isinstance(value, bool) and isinstance(value, (int, Integer))


@dataclass(frozen=True)
class CassonGordonSummand:
    r"""Exact contribution of one signed ``(2,q)``-cable summand.

    ``pattern_signature`` and ``companion_signature`` already include the
    sign of the connected-sum component.  The latter is the raw
    Levine--Tristram value; its contribution to the Casson--Gordon signature
    is twice that value, as exposed by ``satellite_signature``.
    """

    component_index: int
    sign: int
    q: object
    character_parameter: object
    pattern_signature: object
    companion_signature: object
    companion_nullity: object

    @property
    def satellite_signature(self):
        """Return the signed ``2*sigma_K(xi_q^a)`` companion contribution."""
        return 2 * self.companion_signature

    @property
    def sigma(self):
        """Return this summand's Casson--Gordon signature."""
        return self.pattern_signature + self.satellite_signature

    @property
    def eta(self):
        """Return this summand's Casson--Gordon nullity."""
        return 2 * self.companion_nullity

    @property
    def is_trivial(self):
        """Return whether the restricted character is trivial."""
        return self.character_parameter == 0


@dataclass(frozen=True)
class CassonGordonInvariant:
    r"""Exact Casson--Gordon data for a signed connected sum.

    The signature is additive.  Nullity satisfies the connected-sum formula:
    the summand nullities add, and joining ``r`` nontrivial character
    restrictions contributes an additional ``r-1`` when ``r > 0``.
    """

    summands: tuple

    @property
    def sigma(self):
        """Return the total Casson--Gordon signature."""
        return sum((summand.sigma for summand in self.summands), QQ(0))

    @property
    def eta(self):
        """Return the total Casson--Gordon nullity."""
        summand_nullity = sum(
            (summand.eta for summand in self.summands),
            Integer(0),
        )
        nontrivial_count = sum(
            1 for summand in self.summands if not summand.is_trivial
        )
        connected_sum_correction = max(nontrivial_count - 1, 0)
        return summand_nullity + connected_sum_correction

    @property
    def pattern_signature(self):
        """Return the sum of all signed torus-pattern contributions."""
        return sum(
            (summand.pattern_signature for summand in self.summands),
            QQ(0),
        )

    @property
    def satellite_signature(self):
        """Return the sum of all signed companion contributions."""
        return sum(
            (summand.satellite_signature for summand in self.summands),
            QQ(0),
        )

    @property
    def character_parameters(self):
        """Return the normalized parameter on every connected-sum summand."""
        return tuple(
            summand.character_parameter for summand in self.summands
        )


def _parse_character_parameters(knot, character_parameters):
    """Return one integer parameter per connected-sum component.

    A scalar is convenient for the overwhelmingly common one-summand case.
    Multiple components require an explicit list or tuple so that component
    order, already observable throughout the structural model, remains clear.
    """
    component_count = len(knot)

    if component_count == 1 and _is_integer(character_parameters):
        parameters = [character_parameters]
    else:
        if not isinstance(character_parameters, (list, tuple)):
            raise TypeError(
                "Character parameters must be an integer for one summand or "
                "a list/tuple with one integer per summand."
            )
        parameters = list(character_parameters)

    if len(parameters) != component_count:
        raise ValueError(
            f"Expected {component_count} character parameters, "
            f"but received {len(parameters)}."
        )

    for component_index, parameter in enumerate(parameters):
        if not _is_integer(parameter):
            raise TypeError(
                "Character parameter at component "
                f"{component_index} must be an integer."
            )

    return parameters


def _outer_two_cable_order(cable_description, component_index):
    """Return ``q`` after validating the outer ``T(2,q)`` pattern.

    In a one-layer torus knot, ``T(2,q)`` and ``T(q,2)`` are isotopic, so both
    descriptions identify the same pattern calculation.  In a multi-layer
    cable, however, the first entry is its winding number; swapping the pair
    would describe a different satellite operation.
    """
    outer_p, outer_q = cable_description[-1]

    if len(cable_description) == 1 and outer_q == 2:
        q = outer_p
    else:
        if outer_p != 2:
            raise ValueError(
                f"Component {component_index} must have outer winding number "
                "2; its outermost cabling pair is "
                f"({outer_p}, {outer_q})."
            )
        q = outer_q

    # The GeneralizedAlgebraicKnot constructor has already established that
    # q > 1 and gcd(2,q) = 1.  Lemma 2.12 additionally assumes primality.
    if not is_prime(q):
        raise ValueError(
            f"Component {component_index} has outer order q={q}; "
            "the Casson--Gordon cabling formula currently requires q prime."
        )

    return Integer(q)


def _summand_invariant(component_index, sign, cable_description, parameter):
    """Compute one already structurally validated signed summand."""
    q = _outer_two_cable_order(cable_description, component_index)
    a = Integer(parameter) % q

    # The trivial character is a separate part of Lemma 2.12.  Substituting
    # a=0 into the displayed nontrivial formula would incorrectly return -q.
    if a == 0:
        return CassonGordonSummand(
            component_index=component_index,
            sign=sign,
            q=q,
            character_parameter=a,
            pattern_signature=QQ(0),
            companion_signature=Integer(0),
            companion_nullity=Integer(0),
        )

    pattern_signature = sign * (
        -q + QQ(2 * a * (q - a)) / q
    )

    if len(cable_description) == 1:
        # T(2,q) is the pattern with the unknot as companion.
        companion_signature = Integer(0)
    else:
        # The earlier pairs describe the complete companion from its base knot
        # outward.  Existing LT code applies all inner cabling substitutions,
        # avoiding the incorrect linear layer multiplier in the legacy code.
        companion = LT_signature_iterated_torus_knot(
            cable_description[:-1]
        )
        companion_signature = sign * companion(QQ(a) / q)

    # Since q is prime and a is nonzero, xi_q^a is a primitive prime-order
    # root.  A knot Alexander polynomial cannot vanish there: divisibility by
    # Phi_q would force its value at one to be divisible by Phi_q(1)=q,
    # contradicting Delta_K(1)=+/-1.  Hence the LT nullity, and therefore the
    # individual cable nullity, is zero throughout this module's domain.
    companion_nullity = Integer(0)

    return CassonGordonSummand(
        component_index=component_index,
        sign=sign,
        q=q,
        character_parameter=a,
        pattern_signature=pattern_signature,
        companion_signature=companion_signature,
        companion_nullity=companion_nullity,
    )


def casson_gordon_invariant(knot, character_parameters):
    r"""Compute Casson--Gordon ``sigma`` and ``eta`` for supported GA-knots.

    Args:
        knot: A ``GeneralizedAlgebraicKnot`` whose components are ordinary
            ``T(2,q)`` knots or iterated torus knots with outermost cabling
            pair ``(2,q)``, where every such ``q`` is prime.
        character_parameters: For one component, an integer ``a`` or a
            one-element sequence.  For a connected sum, a list/tuple with one
            integer per component.  Each parameter is reduced modulo its
            component's outer prime ``q``.

    Returns:
        A ``CassonGordonInvariant`` with an immutable contribution record for
        every connected-sum component.

    Raises:
        TypeError: If ``knot`` or a character parameter has the wrong type.
        ValueError: If the parameter count is wrong or a component lies
            outside the currently implemented ``(2,q)``, prime-``q`` domain.
    """
    if not isinstance(knot, GeneralizedAlgebraicKnot):
        raise TypeError(
            "Expected a GeneralizedAlgebraicKnot object, "
            f"got {type(knot)}."
        )

    parameters = _parse_character_parameters(knot, character_parameters)
    summands = []

    for component_index, ((sign, cable_description), parameter) in enumerate(
        zip(knot.description, parameters)
    ):
        summands.append(
            _summand_invariant(
                component_index,
                sign,
                cable_description,
                parameter,
            )
        )

    return CassonGordonInvariant(tuple(summands))
