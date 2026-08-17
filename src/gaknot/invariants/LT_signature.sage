#!/usr/bin/env sage -python

r"""Levine--Tristram signatures for the knot descriptions used by gaknot.

The signature functions in this module are represented by
``SignatureFunction`` objects.  A ``SignatureFunction`` stores a ``Counter``
whose keys are exact points in ``[0, 1)`` and whose values are jump weights.
Crossing a point of weight ``j`` changes the signature by ``2*j``; at the
discontinuity itself the class uses the midpoint value.

There are three levels of input:

* ``(p, q)`` describes the positive torus knot ``T(p, q)``;
* ``[(p_1, q_1), ..., (p_n, q_n)]`` describes an iterated torus knot, starting
  with the innermost knot and listing successive outer cables;
* ``[(sign, knot_description), ...]`` describes a signed connected sum of
  iterated torus knots, where every sign is either ``1`` or ``-1``.

All computations use Sage rational numbers for jump locations.  Keeping the
locations exact is essential: two mathematically equal jumps must become the
same ``Counter`` key so that cancellation works reliably.
"""

import math
from collections import Counter

from gaknot.invariants.signature import SignatureFunction

from sage.all import Integer, inverse_mod, floor


# ---------------------------------------------------------------------------
# Validation and the elementary torus-knot jump formula
# ---------------------------------------------------------------------------

def _validate_torus_knot_parameters(p, q):
    """Validate the two integer parameters defining a nontrivial torus knot.

    Python booleans need an explicit check because ``bool`` is a subclass of
    ``int``.  Treating ``True`` as the cabling parameter ``1`` would otherwise
    produce a misleading positivity error instead of a type error.
    """
    # Accept both ordinary Python integers and Sage's exact Integer objects.
    if (
        isinstance(p, bool)
        or isinstance(q, bool)
        or not isinstance(p, (int, Integer))
        or not isinstance(q, (int, Integer))
    ):
        raise TypeError('Parameters p and q have to be integers.')

    # Parameters 0, 1, and negative integers do not define the nontrivial
    # positive torus knots supported by this API.
    if p <= 1 or q <= 1:
        raise ValueError('Parameters p and q must be >1.')

    # Coprimality makes T(p, q) a knot rather than a multi-component torus link.
    # It also guarantees that the modular inverse used below exists.
    if math.gcd(p, q) != 1:
        raise ValueError('Parameters p and q must be relatively prime.')


def _torus_knot_signature_counter(p, q):
    """Return the signature-jump counter for already validated ``p`` and ``q``.

    Validation deliberately lives at the public-entry-point level.  This
    internal helper is shared by the torus and iterated computations so that
    the modular-arithmetic formula has only one implementation.
    """
    # Since gcd(p, q) = 1, p has an inverse modulo q.  It lets us recover the
    # coefficient a in i = a*p + b*q without searching over lattice points.
    p_inv_q = inverse_mod(p, q)
    counter = Counter()

    # Candidate roots lie at x = i/(p*q).  Multiples of p or q are excluded:
    # at those locations one of q*x or p*x is integral and there is no jump.
    for i in range(1, p * q):
        if i % p == 0 or i % q == 0:
            continue

        # Choose the unique representative 0 <= a < q in i = a*p + b*q.
        a_val = (i * p_inv_q) % q
        b_val = (i - a_val * p) // q

        # The usual lattice-point formula contains floor(a/q), but that term
        # is zero for the chosen representative.  The parity of the remaining
        # floors determines whether the jump weight is +1 or -1.
        exponent = floor(b_val / p) + floor(a_val / q + b_val / p)

        # Integer(i)/(p*q) is a Sage rational, not a floating-point number.
        # Exact keys are necessary when several summands contribute at the
        # same jump location.
        counter[Integer(i) / (p * q)] = (-1) ** exponent

    return counter


def _validate_iterated_torus_knot_description(desc):
    """Validate every cabling pair in an iterated-knot description.

    Errors are annotated with the pair's index so that a bad outer cabling
    parameter can be located in a long description.  The original exception
    category is preserved: container/number type problems remain ``TypeError``
    and mathematically invalid integer values remain ``ValueError``.
    """
    # Lists are the canonical representation, while tuples are accepted for
    # callers who prefer immutable descriptions.
    if not isinstance(desc, (list, tuple)):
        raise TypeError('The variable desc should be a list or tuple.')

    # An empty sequence has no base torus knot and therefore cannot describe
    # an iterated torus knot.
    if len(desc) == 0:
        raise ValueError('The description must contain at least one cabling pair.')

    for i, element in enumerate(desc):
        # Check the pair shape before unpacking it, avoiding low-level unpacking
        # errors with little information about the offending layer.
        if not isinstance(element, (list, tuple)) or len(element) != 2:
            raise ValueError(
                f'Cable parameter at index {i} must be a pair (p, q).'
            )

        p, q = element
        try:
            # Every cable obeys exactly the same numerical contract as a
            # standalone torus knot.
            _validate_torus_knot_parameters(p, q)
        except TypeError as error:
            raise TypeError(
                f'Invalid cable parameter at index {i}: {error}'
            ) from error
        except ValueError as error:
            raise ValueError(
                f'Invalid cable parameter at index {i}: {error}'
            ) from error


def _validate_generalized_algebraic_knot_description(desc):
    """Validate a signed connected sum of iterated torus knots.

    Each outer element must have the form ``(sign, knot_desc)``.  Nested
    validation errors receive the summand index in addition to any cabling
    index already supplied by the iterated validator.
    """
    if not isinstance(desc, (list, tuple)):
        raise TypeError('The variable desc should be a list or tuple.')

    # The empty connected-sum description is intentionally not used as an
    # encoding of the unknot; every public description must contain a summand.
    if len(desc) == 0:
        raise ValueError('The description must contain at least one summand.')

    for i, element in enumerate(desc):
        # Validate structure before unpacking for a stable, contextual error.
        if not isinstance(element, (list, tuple)) or len(element) != 2:
            raise ValueError(
                f'Element at index {i} must be a pair '
                '(sign, knot_description).'
            )

        sign, knot_desc = element
        # Exclude booleans explicitly because True == 1 and False == 0.
        # The sign records whether the summand is added or concordance-inverted.
        if isinstance(sign, bool) or sign not in (1, -1):
            raise ValueError(f'Sign at index {i} must be 1 or -1.')

        try:
            # Delegate the nested format and arithmetic checks to the same
            # validator used by the standalone iterated-signature API.
            _validate_iterated_torus_knot_description(knot_desc)
        except TypeError as error:
            raise TypeError(
                f'Invalid knot description at index {i}: {error}'
            ) from error
        except ValueError as error:
            raise ValueError(
                f'Invalid knot description at index {i}: {error}'
            ) from error


# ---------------------------------------------------------------------------
# Public signature constructors
# ---------------------------------------------------------------------------

def LT_signature_torus_knot(p, q):
    """Compute the Levine--Tristram signature of the torus knot ``T(p, q)``.

    The modular-arithmetic implementation visits the ``p*q - 1`` possible
    rational locations once, so its running time is ``O(p*q)``.

    The returned function uses exact rational jump locations and the midpoint
    convention at discontinuities implemented by ``SignatureFunction``.
    """
    # Validate before modular inversion so callers receive an API-level error
    # instead of a low-level arithmetic exception.
    _validate_torus_knot_parameters(p, q)
    return SignatureFunction(counter=_torus_knot_signature_counter(p, q))


def reparametrize(sig_func, p):
    r"""Return the reparametrized function ``theta -> sig_func(p*theta)``.

    A jump at ``x`` has one preimage in every interval ``[k/p, (k+1)/p)``.
    Thus its preimages are ``(x+k)/p`` for ``k = 0, ..., p-1`` and all retain
    the original jump weight.  This is the reparametrization appearing in the
    cable-signature formula.
    """
    # Read the sparse jump representation directly; sampling function values
    # would lose the exact discontinuity information.
    old_counter = sig_func.jumps_counter
    new_counter = Counter()

    for x, jump_val in old_counter.items():
        # Multiplication of the argument by p wraps around the unit interval p
        # times, so each old jump gives rise to p new jumps.
        for k in range(p):
            new_x = (x + k) / p
            # Use += because different contributions may meet and cancel at an
            # identical exact rational location.
            new_counter[new_x] += jump_val

    return SignatureFunction(counter=new_counter)


def LT_signature_iterated_torus_knot_counter(desc):
    r"""Return the combined jump counter of an iterated torus knot.

    If ``desc = [(p_1,q_1), ..., (p_n,q_n)]``, the first pair is the innermost
    torus knot and later pairs are successive cables.  The cable formula is

    ``sigma_{K_{p,q}}(theta) = sigma_K(p*theta) + sigma_{T(p,q)}(theta)``.

    Consequently the component at index ``i`` is reparametrized by the product
    of the ``p`` parameters belonging to all outer layers ``j > i``.  Returning
    a ``Counter`` lets generalized connected sums combine sparse jumps without
    repeatedly constructing temporary ``SignatureFunction`` objects.
    """
    # Validate the entire input first.  No partial counter is returned if a
    # later layer is malformed.
    _validate_iterated_torus_knot_description(desc)

    total_counter = Counter()
    # This is the product of p-values in the outer layers already processed.
    current_p_prod = 1

    # Work from the outermost cable back toward the innermost torus knot so the
    # required winding-number product is available incrementally.
    for i in range(len(desc) - 1, -1, -1):
        p, q = desc[i]
        # Reuse the elementary torus formula rather than maintaining a second
        # copy of its modular arithmetic here.
        component_counter = _torus_knot_signature_counter(p, q)

        for base_jump, jump_val in component_counter.items():
            if current_p_prod == 1:
                # The outermost pattern is not reparametrized.
                total_counter[base_jump] += jump_val
            else:
                # A jump x in sigma(theta) becomes a jump at (x+k)/w in
                # sigma(w*theta), for k = 0, ..., w-1.
                for k in range(current_p_prod):
                    total_counter[(base_jump + k) / current_p_prod] += jump_val

        # Inner components see this cabling winding number as well as every
        # winding number accumulated from still farther outside.
        current_p_prod *= p

    return total_counter


def LT_signature_iterated_torus_knot(desc):
    r"""Compute the Levine--Tristram signature of an iterated torus knot.

    See ``LT_signature_iterated_torus_knot_counter`` for the description order
    and cable formula.  This wrapper converts the accumulated sparse counter to
    the public ``SignatureFunction`` representation.
    """
    counter = LT_signature_iterated_torus_knot_counter(desc)
    return SignatureFunction(counter=counter)


def LT_signature_generalized_algebraic_knot(desc):
    """Compute the signature of a signed sum of iterated torus knots.

    Signature is additive under connected sum and changes sign under the
    concordance inverse.  Therefore a summand tagged by ``1`` contributes its
    jump counter and a summand tagged by ``-1`` subtracts that counter.

    Accumulating counters first avoids creating a temporary signature function
    after every addition and allows exact cancellations to happen immediately.
    """
    # Complete validation precedes computation, giving consistent error types
    # and index context even when an invalid summand occurs late in the list.
    _validate_generalized_algebraic_knot_description(desc)

    total_counter = Counter()

    for sign, knot_desc in desc:
        # Obtain sparse component data directly rather than constructing and
        # immediately unpacking an intermediate SignatureFunction.
        component_counter = LT_signature_iterated_torus_knot_counter(knot_desc)

        if sign == 1:
            # Counter.update performs coefficient-wise addition.
            total_counter.update(component_counter)
        else:
            # Counter.subtract performs coefficient-wise subtraction and keeps
            # negative jump weights, exactly as required for the inverse knot.
            total_counter.subtract(component_counter)

    # SignatureFunction removes any zero entries created by cancellation and
    # precomputes its sorted jump data for efficient later evaluation.
    return SignatureFunction(counter=total_counter)
