#!/usr/bin/env sage -python

r"""End-to-end metabelian signature jumps for common-``p`` iterated cables.

This module joins three calculations that are intentionally kept independent
elsewhere in the package:

1. the character on the outer torus-pattern summand of branched-cover
   homology is converted from Smith coordinates to its deck orbit;
2. Yanagida's formulas compute all currently accessible twisted jumps of the
   outer pattern; and
3. Theorem 4.19 adds the ordinary Levine--Tristram signatures of the inner
   companion at the character-dependent phase shifts.

The supported knot has one positive iterated-torus summand

``T(p,q_1; p,q_2; ...; p,q_l)``,

where the sequence is written from the innermost torus knot to the outermost
cabling pattern.  The character must live on its ``p``-fold branched cover.
At the outer satellite stage the winding is also ``p``, so the cover degree
divides the winding and Theorem 4.19 gives exactly ``p`` *ordinary* companion
forms.  If the orbit is ``a_0,...,a_{p-1}`` for the outer ``T(p,q_l)`` pattern,
the result is

``Bl_twisted(T(p,q_l), chi)``
``  + direct_sum_i Bl(companion)(zeta_(q_l)^a_i * t)``.

The result remains coverage-aware.  Yanagida's theorems exclude ``t=1`` and
may exclude positive-dimensional exceptional primary modules.  Those roots
are preserved as explicit gaps in the returned total profile; this function
does not replace an unknown local contribution by zero.
"""

from dataclasses import dataclass, field

from gaknot.core.gaknot import GeneralizedAlgebraicKnot
from gaknot.invariants.LT_signature import LT_signature_iterated_torus_knot
from gaknot.invariants.character import Character
from gaknot.invariants.metabelian_satellite import (
    Theorem419SignatureResult,
    theorem_4_19_signature_jumps,
)
from gaknot.invariants.signature import SignatureFunction
from gaknot.invariants.torus_character import (
    TorusCharacterOrbit,
    torus_character_orbit,
)
from gaknot.invariants.torus_twisted_blanchfield import YanagidaTorusData
from gaknot.invariants.torus_twisted_signature import (
    YanagidaSignatureProfile,
    yanagida_signature_profile,
)


@dataclass(frozen=True)
class IteratedTorusMetabelianSignatureResult:
    r"""Inspectable output of the supported common-``p`` computation.

    ``cable_sequence`` records the knot from inner to outer.  ``orbit`` exposes
    both the Smith-coordinate character and the integer deck orbit used by the
    formulas.  ``yanagida_profile`` retains the root-by-root coverage analysis
    of the outer pattern.  ``companion_signature`` is the ordinary signature
    of every layer below the outermost one, and ``satellite_result`` records
    the assembled Theorem 4.19 summands and total.

    The expanded record is deliberate: when a total jump is unresolved, a
    caller can determine whether the gap came from Yanagida's pattern formula
    and can inspect the exact orbit that selected the exceptional root.
    """

    cable_sequence: tuple
    orbit: TorusCharacterOrbit
    yanagida_profile: YanagidaSignatureProfile = field(repr=False)
    companion_signature: SignatureFunction = field(repr=False)
    satellite_result: Theorem419SignatureResult = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.cable_sequence, tuple) or not self.cable_sequence:
            raise ValueError("cable_sequence must be a nonempty tuple.")
        if any(
            not isinstance(pair, tuple) or len(pair) != 2
            for pair in self.cable_sequence
        ):
            raise TypeError("Every cable_sequence entry must be a (p,q) tuple.")
        if not isinstance(self.orbit, TorusCharacterOrbit):
            raise TypeError("orbit must be a TorusCharacterOrbit object.")
        if not isinstance(self.yanagida_profile, YanagidaSignatureProfile):
            raise TypeError(
                "yanagida_profile must be a YanagidaSignatureProfile object."
            )
        if not isinstance(self.companion_signature, SignatureFunction):
            raise TypeError("companion_signature must be a SignatureFunction object.")
        if not isinstance(self.satellite_result, Theorem419SignatureResult):
            raise TypeError(
                "satellite_result must be a Theorem419SignatureResult object."
            )

        common_p = self.cable_sequence[0][0]
        if any(pair[0] != common_p for pair in self.cable_sequence):
            raise ValueError("Every cable layer must have the same p parameter.")
        outer_q = self.cable_sequence[-1][1]

        # Tie every diagnostic object to the same mathematical input.  These
        # checks make it impossible to manually assemble a result whose orbit,
        # Yanagida matrices, and satellite metadata describe different knots.
        if self.orbit.p != common_p or self.orbit.q != outer_q:
            raise ValueError("orbit does not describe the outer torus pattern.")
        data = self.yanagida_profile.data
        if (
            data.m != self.orbit.p
            or data.n != self.orbit.q
            or data.b != self.orbit.a_values
        ):
            raise ValueError("yanagida_profile does not use the recorded orbit.")
        if (
            self.satellite_result.cover_degree != common_p
            or self.satellite_result.winding != common_p
            or self.satellite_result.case != "ordinary_companion"
        ):
            raise ValueError(
                "satellite_result does not describe the common-p divisible branch."
            )
        if self.satellite_result.phase_arguments != self.orbit.phase_arguments:
            raise ValueError("satellite phases do not match the character orbit.")

    @property
    def pattern_profile(self):
        """Return the coverage-aware twisted profile of the outer pattern."""
        return self.satellite_result.pattern_profile

    @property
    def companion_summands(self):
        """Return the ``p`` shifted ordinary companion jump profiles."""
        return self.satellite_result.companion_summands

    @property
    def total_profile(self):
        """Return the assembled known jumps and all propagated coverage gaps."""
        return self.satellite_result.total_profile

    @property
    def is_complete(self):
        """Return whether every jump in the assembled satellite is known."""
        return self.satellite_result.is_complete

    @property
    def unresolved_arguments(self):
        """Return the roots at which some local contribution remains unknown."""
        return self.satellite_result.unresolved_arguments


def iterated_torus_metabelian_signature_jumps(knot, character):
    r"""Compute twisted signature jumps for a positive common-``p`` cable.

    Args:
        knot: A one-summand positive :class:`GeneralizedAlgebraicKnot` whose
            cable sequence is ``[(p,q_1),...,(p,q_l)]``.
        character: A :class:`Character` on the ``p``-fold branched-cover
            homology of the same structural knot.

    Returns:
        An :class:`IteratedTorusMetabelianSignatureResult` exposing the
        character orbit, Yanagida pattern calculation, ordinary companion
        signature, shifted summands, and coverage-aware total profile.

    Raises:
        TypeError: If either public argument has the wrong type.
        ValueError: If the character belongs to another knot or cover.
        NotImplementedError: If the knot is negative, a connected sum, or has
            nonconstant first cabling parameters.  Those families require
            orientation bookkeeping or recursive induced-character transport
            beyond the currently implemented end-to-end path.

    Mathematical route:

    * the outer homology layer is ``H_1(Sigma_p(T(p,q_l)))`` with one copy;
    * its Smith-coordinate character determines ``a_0,...,a_{p-1}``;
    * ``YanagidaTorusData(p,q_l,a)`` supplies the pattern profile;
    * the inner sequence has its ordinary Levine--Tristram signature; and
    * Theorem 4.19 is called with cover degree ``p``, winding ``p``, phase
      arguments ``a_i/q_l``, and exponent ``p/p=1``.
    """
    if not isinstance(knot, GeneralizedAlgebraicKnot):
        raise TypeError("knot must be a GeneralizedAlgebraicKnot object.")
    if not isinstance(character, Character):
        raise TypeError("character must be a Character object.")
    if not knot.is_iterated_torus_knot():
        raise NotImplementedError(
            "The end-to-end formula currently supports one positive "
            "iterated torus knot."
        )

    _, cable_sequence_list = knot.description[0]
    cable_sequence = tuple(tuple(pair) for pair in cable_sequence_list)
    common_p = cable_sequence[0][0]
    if any(pair[0] != common_p for pair in cable_sequence):
        raise NotImplementedError(
            "The end-to-end formula currently requires the same p parameter "
            "in every cable layer."
        )

    homology = character.homology
    if homology.knot.description != knot.description:
        raise ValueError(
            "Character must be defined on the homology of the supplied knot."
        )
    if homology.cover_degree != common_p:
        raise ValueError(
            f"Formula requires a character on the {int(common_p)}-fold cover "
            f"of the common-p cable. Got N={int(homology.cover_degree)}."
        )

    # BranchedCoverHomology lists satellite layers outside-in, the reverse of
    # the knot description.  In the p-fold cover the outer T(p,q_l) layer has
    # effective cover degree p and multiplicity one.  After this stage the
    # companion cover degree is p/gcd(p,p)=1, explaining why every deeper layer
    # has trivial cover homology and why Theorem 4.19 uses ordinary signatures.
    component = homology.decomposition[0]
    outer_layer = component["layers"][0]
    outer_q = cable_sequence[-1][1]
    if (
        outer_layer["cable_index"] != len(cable_sequence) - 1
        or tuple(outer_layer["parameters"]) != (common_p, outer_q)
        or outer_layer["effective_N"] != common_p
        or outer_layer["multiplicity"] != 1
    ):
        raise ArithmeticError(
            "Branched-cover decomposition does not contain the expected "
            "outer common-p layer."
        )

    outer_copies = character.restrict_to_layer(0, 0)
    if len(outer_copies) != 1:
        raise ArithmeticError("The outer common-p layer must have one copy.")
    orbit = torus_character_orbit(common_p, outer_q, outer_copies[0])

    # The exact same integer tuple plays the role of Yanagida's b-vector and of
    # the phase exponent vector in Theorem 4.19.  No cyclic reordering is
    # inserted between these consumers.
    yanagida_data = YanagidaTorusData(
        common_p,
        outer_q,
        orbit.a_values,
    )
    yanagida_profile = yanagida_signature_profile(yanagida_data)

    # Removing the final pair leaves the complete inner companion.  For a
    # one-layer torus knot that companion is the unknot, whose signature has no
    # jumps; the LT helper deliberately requires a nonempty knot description,
    # so construct the empty SignatureFunction directly in that case.
    companion_sequence = cable_sequence[:-1]
    if companion_sequence:
        companion_signature = LT_signature_iterated_torus_knot(
            companion_sequence
        )
    else:
        companion_signature = SignatureFunction()

    satellite_result = theorem_4_19_signature_jumps(
        yanagida_profile,
        common_p,
        common_p,
        phase_arguments=orbit.phase_arguments,
        ordinary_companion_signature=companion_signature,
    )

    return IteratedTorusMetabelianSignatureResult(
        cable_sequence=cable_sequence,
        orbit=orbit,
        yanagida_profile=yanagida_profile,
        companion_signature=companion_signature,
        satellite_result=satellite_result,
    )
