#!/usr/bin/env sage -python

r"""End-to-end metabelian signatures for iterated torus cables.

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

For a nontrivial prime-power-order character, Theorem 4.14 of
Borodzik--Conway--Politarczyk II proves that the metabelian Blanchfield pairing
is representable.  If the jump computation has no gap away from ``t=1``,
:func:`iterated_torus_metabelian_signature_function` uses representability to
recover the missing jump at one, normalizes the averaged signature to vanish
there, and exposes the corresponding Casson--Gordon signature difference.

The module also implements one bounded step of Theorem 4.19's nondivisible
branch.  When the outer cover degree does not divide the outer winding,
:func:`iterated_torus_nondivisible_signature_jumps` transports the source
character to the ``h`` lower-cover companion characters, computes every
companion profile when the remaining cable is in the common-``p`` domain, and
assembles the phase/power substitutions prescribed by the theorem.  Its outer
pattern profile is explicit caller input.  This is essential: the presently
implemented Yanagida matrices do not supply the required pattern form when
the pattern winding and representation dimension differ.
"""

from dataclasses import dataclass, field

from sage.all import Integer

from gaknot.core.gaknot import GeneralizedAlgebraicKnot
from gaknot.invariants.LT_signature import LT_signature_iterated_torus_knot
from gaknot.invariants.character import Character
from gaknot.invariants.character_transport import InducedCompanionCharacters
from gaknot.invariants.metabelian_satellite import (
    AveragedTwistedSignatureFunction,
    Theorem419SignatureResult,
    TwistedSignatureJumpProfile,
    averaged_signature_from_representable_profile,
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


@dataclass(frozen=True)
class NondivisibleIteratedTorusMetabelianSignatureResult:
    r"""Inspectable assembly of Theorem 4.19's nondivisible branch.

    ``transport`` contains the ``h`` induced characters and their equally
    ordered phase arguments.  ``companion_results[j]`` is the complete
    coverage-aware calculation performed on the inner companion with
    ``transport.characters[j]``.  Finally, ``satellite_result`` substitutes
    those ``h`` raw profiles by the corresponding phase and power and adds
    them to the caller-supplied outer ``pattern_profile``.

    The outer profile is explicit input because Yanagida's current formulas
    calculate the cover-degree-``m`` representation of ``T(m,n)``.  At a
    nondivisible ``(p,q)`` cabling stage the satellite cover degree is not
    ``p``, so manufacturing a Yanagida profile there would assert a formula
    outside its proved domain.
    """

    cable_sequence: tuple
    transport: InducedCompanionCharacters = field(repr=False)
    pattern_profile: TwistedSignatureJumpProfile = field(repr=False)
    companion_results: tuple = field(repr=False)
    satellite_result: Theorem419SignatureResult = field(repr=False)

    def __post_init__(self):
        if (
            not isinstance(self.cable_sequence, tuple)
            or len(self.cable_sequence) < 2
        ):
            raise ValueError(
                "cable_sequence must contain an outer pattern and a "
                "nontrivial companion."
            )
        if any(
            not isinstance(pair, tuple) or len(pair) != 2
            for pair in self.cable_sequence
        ):
            raise TypeError("Every cable_sequence entry must be a (p,q) tuple.")
        if not isinstance(self.transport, InducedCompanionCharacters):
            raise TypeError(
                "transport must be an InducedCompanionCharacters object."
            )
        if not isinstance(
            self.pattern_profile,
            TwistedSignatureJumpProfile,
        ):
            raise TypeError(
                "pattern_profile must be a TwistedSignatureJumpProfile object."
            )
        if not isinstance(self.companion_results, tuple) or any(
            not isinstance(item, IteratedTorusMetabelianSignatureResult)
            for item in self.companion_results
        ):
            raise TypeError(
                "companion_results must be a tuple of "
                "IteratedTorusMetabelianSignatureResult objects."
            )
        if not isinstance(self.satellite_result, Theorem419SignatureResult):
            raise TypeError(
                "satellite_result must be a Theorem419SignatureResult object."
            )

        source_description = (
            self.transport.source_character.homology.knot.description
        )
        if source_description != [(1, list(self.cable_sequence))]:
            raise ValueError(
                "cable_sequence does not describe the transport's source knot."
            )
        companion_sequence = self.cable_sequence[:-1]
        if self.transport.companion_knot.description != [
            (1, list(companion_sequence))
        ]:
            raise ValueError(
                "transport does not remove the recorded outer pattern."
            )
        if len(self.companion_results) != self.transport.h:
            raise ValueError(
                "There must be one companion result per induced character."
            )
        for result in self.companion_results:
            if result.cable_sequence != companion_sequence:
                raise ValueError(
                    "A companion result describes the wrong inner cable."
                )
            if (
                result.satellite_result.cover_degree
                != self.transport.lower_cover_degree
            ):
                raise ValueError(
                    "A companion result uses the wrong lower cover degree."
                )

        if self.pattern_profile.cover_degree != self.transport.cover_degree:
            raise ValueError(
                "pattern_profile must carry the satellite cover degree."
            )
        theorem_result = self.satellite_result
        if (
            theorem_result.case != "metabelian_companion"
            or theorem_result.cover_degree != self.transport.cover_degree
            or theorem_result.winding != self.transport.winding
            or theorem_result.h != self.transport.h
        ):
            raise ValueError(
                "satellite_result does not describe this nondivisible stage."
            )
        if theorem_result.phase_arguments != self.transport.phase_arguments:
            raise ValueError(
                "satellite phases do not match the transported phase orbit."
            )
        if theorem_result.pattern_profile != self.pattern_profile:
            raise ValueError(
                "satellite_result does not use the recorded pattern profile."
            )

        # The profiles computed from the induced characters must be exactly
        # the profiles transformed by Theorem 4.19.  Checking each summand
        # independently catches a phase-order permutation even when the final
        # direct sum happens to be unchanged by that permutation.
        exponent = self.transport.winding // self.transport.h
        for index, (result, phase, summand) in enumerate(zip(
            self.companion_results,
            self.transport.phase_arguments,
            theorem_result.companion_summands,
        )):
            expected = result.total_profile.pullback(
                phase,
                exponent,
                label=f"metabelian companion summand {index + 1}",
            )
            if summand != expected:
                raise ValueError(
                    "satellite_result does not preserve the ordered "
                    "character/profile/phase correspondence."
                )

    @property
    def induced_characters(self):
        """Return the ``h`` lower-cover characters in theorem order."""

        return self.transport.characters

    @property
    def phase_arguments(self):
        """Return the phase paired with each induced character."""

        return self.transport.phase_arguments

    @property
    def companion_profiles(self):
        """Return the raw lower-cover profile computed for every character."""

        return tuple(result.total_profile for result in self.companion_results)

    @property
    def companion_summands(self):
        """Return the phase/power-pulled companion contributions."""

        return self.satellite_result.companion_summands

    @property
    def total_profile(self):
        """Return the assembled pattern and transformed-companion profile."""

        return self.satellite_result.total_profile

    @property
    def is_complete(self):
        """Return whether every jump in the assembled satellite is known."""

        return self.satellite_result.is_complete

    @property
    def unresolved_arguments(self):
        """Return roots retaining at least one unresolved local contribution."""

        return self.satellite_result.unresolved_arguments


def _character_order(character):
    r"""Return the order of the image of ``character`` in ``Q/Z``.

    A reduced rational value ``a/d`` has order ``d`` in ``Q/Z``.  The image
    of the whole character is generated by all coordinate values, so its
    order is the least common multiple of their denominators.  Empty and zero
    coordinate lists therefore give order one, i.e. the trivial character.
    """
    order = Integer(1)
    for value in character.values:
        order = order.lcm(Integer(value.denominator()))
    return order


@dataclass(frozen=True)
class IteratedTorusMetabelianSignatureFunctionResult:
    r"""A normalized twisted signature and its complete calculation record.

    ``jump_result`` retains the character orbit, Yanagida local calculations,
    ordinary companion signature, and original coverage gaps.  In particular,
    it shows that any gap completed by representability was located at
    ``t=1``.  ``character_order`` records the prime power used to invoke
    Theorem 4.14.  ``signature_function`` contains the completed jump profile
    and evaluates the globally normalized averaged twisted signature.

    Calling the result delegates to ``signature_function``.  The method
    :meth:`casson_gordon_signature_difference_at` returns

    ``sign_av_omega(tau(K,chi)) - sign_av_1(tau(K,chi))``

    by negating that value, exactly as prescribed by Theorem 4.14(b).
    """

    jump_result: IteratedTorusMetabelianSignatureResult = field(repr=False)
    character_order: object
    signature_function: AveragedTwistedSignatureFunction = field(repr=False)

    def __post_init__(self):
        if not isinstance(
            self.jump_result,
            IteratedTorusMetabelianSignatureResult,
        ):
            raise TypeError(
                "jump_result must be an "
                "IteratedTorusMetabelianSignatureResult object."
            )
        if isinstance(self.character_order, bool) or not isinstance(
            self.character_order,
            (int, Integer),
        ):
            raise TypeError("character_order must be an integer.")
        character_order = Integer(self.character_order)
        if character_order <= 1 or not character_order.is_prime_power():
            raise ValueError(
                "character_order must be a nontrivial prime power."
            )
        if not isinstance(
            self.signature_function,
            AveragedTwistedSignatureFunction,
        ):
            raise TypeError(
                "signature_function must be an "
                "AveragedTwistedSignatureFunction object."
            )

        source_profile = self.jump_result.total_profile
        nontrivial_gaps = tuple(
            gap for gap in source_profile.unresolved if gap.argument != 0
        )
        if nontrivial_gaps:
            raise ValueError(
                "jump_result still has unresolved nontrivial-root jumps."
            )
        if (
            self.signature_function.jump_profile.cover_degree
            != source_profile.cover_degree
        ):
            raise ValueError(
                "signature_function and jump_result use different cover degrees."
            )

        # The completed function must preserve every contribution already
        # proved by Yanagida and Theorem 4.19.  Only the aggregate jump at
        # argument zero may have been added by representability.
        source_nontrivial = {
            argument: weight
            for argument, weight in source_profile.known_jumps
            if argument != 0
        }
        completed_nontrivial = {
            argument: weight
            for argument, weight
            in self.signature_function.jump_profile.known_jumps
            if argument != 0
        }
        if source_nontrivial != completed_nontrivial:
            raise ValueError(
                "signature_function changed a proved nontrivial-root jump."
            )

        object.__setattr__(self, "character_order", character_order)

    @property
    def cable_sequence(self):
        """Return the inner-to-outer cable sequence from the jump result."""
        return self.jump_result.cable_sequence

    @property
    def orbit(self):
        """Return the exact outer character orbit used by both formulas."""
        return self.jump_result.orbit

    @property
    def jump_profile(self):
        """Return the completed, zero-total-jump profile."""
        return self.signature_function.jump_profile

    def __call__(self, argument):
        """Evaluate the normalized averaged twisted signature."""
        return self.signature_function(argument)

    def casson_gordon_signature_difference_at(self, argument):
        r"""Return the Casson--Gordon signature difference at ``argument``."""
        return -self.signature_function(argument)


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


def iterated_torus_nondivisible_signature_jumps(
    knot,
    character,
    pattern_profile,
):
    r"""Assemble one nondivisible cabling stage from concrete character data.

    The function removes the outermost cable, computes the induced characters
    and phases, evaluates the existing common-``p`` end-to-end formula on the
    inner companion for every induced character, and passes the resulting raw
    profiles to Theorem 4.19.

    Args:
        knot: A positive one-summand iterated torus knot with at least two
            cabling layers.
        character: A :class:`Character` on the branched cover of ``knot``.
            Its cover degree ``n`` and outer winding ``w`` must satisfy
            ``n`` not dividing ``w``.
        pattern_profile: A coverage-aware
            :class:`TwistedSignatureJumpProfile` for the outer torus pattern
            in the original ``n``-dimensional representation.  The profile
            must declare ``cover_degree=n``.

    Returns:
        A :class:`NondivisibleIteratedTorusMetabelianSignatureResult` retaining
        the induced characters, exact phases, each independently computed
        lower-cover result, the transformed companion summands, and the total
        coverage-aware profile.

    Raises:
        TypeError: If a public argument has the wrong type.
        ValueError: If the character belongs to another knot, the selected
            stage is divisible, or the supplied pattern profile has the wrong
            cover degree.
        NotImplementedError: If the inner companion does not have a common
            first cabling parameter equal to its induced cover degree.  In
            that case its twisted profile requires a further nondivisible
            recursion or another pattern calculation not yet supplied by this
            bounded API.

    The requirement that ``pattern_profile`` be supplied is deliberate.
    Yanagida's explicit matrices describe an ``m``-dimensional representation
    of ``T(m,q)``.  Here the outer pattern is ``T(w,q)`` but nondivisibility
    forces the original cover degree away from the currently supported
    ``m=w`` situation.  The function therefore never substitutes an empty
    profile for an unknown outer contribution.
    """

    if not isinstance(knot, GeneralizedAlgebraicKnot):
        raise TypeError("knot must be a GeneralizedAlgebraicKnot object.")
    if not isinstance(character, Character):
        raise TypeError("character must be a Character object.")
    if not isinstance(pattern_profile, TwistedSignatureJumpProfile):
        raise TypeError(
            "pattern_profile must be a TwistedSignatureJumpProfile object."
        )
    if not knot.is_iterated_torus_knot():
        raise NotImplementedError(
            "The nondivisible end-to-end formula currently supports one "
            "positive iterated torus knot."
        )

    _, cable_sequence_list = knot.description[0]
    cable_sequence = tuple(tuple(pair) for pair in cable_sequence_list)
    if len(cable_sequence) < 2:
        raise ValueError(
            "The nondivisible stage requires a nontrivial inner companion."
        )

    homology = character.homology
    if homology.knot.description != knot.description:
        raise ValueError(
            "Character must be defined on the homology of the supplied knot."
        )
    cover_degree = Integer(homology.cover_degree)
    winding = Integer(cable_sequence[-1][0])
    if winding % cover_degree == 0:
        raise ValueError(
            "The outer winding is divisible by the cover degree; use "
            "iterated_torus_metabelian_signature_jumps for the supported "
            "ordinary-companion branch."
        )
    if pattern_profile.cover_degree != cover_degree:
        actual_degree = pattern_profile.cover_degree
        if actual_degree is None:
            actual_text = "no cover degree"
        else:
            actual_text = f"cover_degree={int(actual_degree)}"
        raise ValueError(
            f"pattern_profile has {actual_text}, but the satellite cover "
            f"has degree {int(cover_degree)}."
        )

    transport = character.induced_companion_characters()
    companion_sequence = cable_sequence[:-1]
    companion_p = Integer(companion_sequence[0][0])
    if any(pair[0] != companion_p for pair in companion_sequence):
        raise NotImplementedError(
            "Each induced companion currently requires a common first "
            "cabling parameter."
        )
    if transport.lower_cover_degree != companion_p:
        raise NotImplementedError(
            "The induced companion cover degree must equal its common first "
            f"cabling parameter. Got degree "
            f"{int(transport.lower_cover_degree)} and p={int(companion_p)}."
        )

    # Each character can select different Yanagida matrices and therefore a
    # different collection of known and exceptional jumps.  Compute them
    # independently; reusing the first profile for every copy would erase the
    # very character dependence that the nondivisible branch preserves.
    companion_results = tuple(
        iterated_torus_metabelian_signature_jumps(
            transport.companion_knot,
            induced_character,
        )
        for induced_character in transport.characters
    )
    companion_profiles = tuple(
        result.total_profile for result in companion_results
    )

    satellite_result = theorem_4_19_signature_jumps(
        pattern_profile,
        cover_degree,
        winding,
        phase_arguments=transport.phase_arguments,
        metabelian_companion_profiles=companion_profiles,
    )
    return NondivisibleIteratedTorusMetabelianSignatureResult(
        cable_sequence=cable_sequence,
        transport=transport,
        pattern_profile=pattern_profile,
        companion_results=companion_results,
        satellite_result=satellite_result,
    )


def iterated_torus_metabelian_signature_function(knot, character):
    r"""Compute the normalized averaged Casson--Gordon twisted signature.

    The function first calls
    :func:`iterated_torus_metabelian_signature_jumps`, so it has the same
    common-``p``, one-positive-summand domain.  It then verifies the additional
    hypotheses needed to turn a partial jump profile into a signature
    function:

    * the character is nontrivial and has prime-power order, as required by
      Theorem 4.14 of Borodzik--Conway--Politarczyk II; and
    * Yanagida's formulas and Theorem 4.19 have resolved every jump away from
      ``t=1``.

    Theorem 4.14(a) then proves that the metabelian Blanchfield pairing is
    representable.  Corollary 5.15 of Borodzik--Conway--Politarczyk I forces
    the sum of all signature jumps to vanish, so the only missing jump, at
    ``t=1``, is the negative sum of the known jumps.  Representability also
    gives the canonical normalization ``sigma_av(1)=0``.

    Args:
        knot: A positive common-``p`` iterated torus knot accepted by
            :func:`iterated_torus_metabelian_signature_jumps`.
        character: A nontrivial prime-power-order :class:`Character` on the
            ``p``-fold branched cover of ``knot``.

    Returns:
        An immutable
        :class:`IteratedTorusMetabelianSignatureFunctionResult`.  Calling it at
        an exact rational argument evaluates ``sigma_av`` at the corresponding
        point ``exp(2*pi*i*argument)``.  Its
        ``casson_gordon_signature_difference_at`` method returns
        ``sign_av_omega(tau)-sign_av_1(tau)``.

    Raises:
        TypeError: For the same public type errors as the jump computation.
        ValueError: If the character is trivial or its order is not a prime
            power, in addition to the structural value errors raised by the
            jump computation.
        NotImplementedError: If the knot lies outside the current common-``p``
            domain or if a positive-dimensional exceptional Yanagida module
            leaves a nontrivial-root jump unresolved.
    """
    jump_result = iterated_torus_metabelian_signature_jumps(knot, character)

    character_order = _character_order(character)
    if character_order <= 1:
        raise ValueError(
            "Theorem 4.14 requires a nontrivial character; the supplied "
            "character has order one."
        )
    if not character_order.is_prime_power():
        raise ValueError(
            "Theorem 4.14 requires a prime-power-order character; the "
            f"supplied character has order {int(character_order)}."
        )

    signature_function = averaged_signature_from_representable_profile(
        jump_result.total_profile
    )
    return IteratedTorusMetabelianSignatureFunctionResult(
        jump_result=jump_result,
        character_order=character_order,
        signature_function=signature_function,
    )
