#!/usr/bin/env sage -python

r"""Coverage-aware signature jumps for the metabelian satellite formula.

This module implements the signature-jump consequence of Theorem 4.19 in
Borodzik--Conway--Politarczyk, *Twisted Blanchfield pairings and twisted
signatures II: Relation to Casson--Gordon invariants*.  The theorem describes
the metabelian Blanchfield pairing of a satellite ``P(K, eta)`` as a direct sum
of a pattern form and explicitly reparametrized companion forms.

Why this module first stores jumps
----------------------------------

The Witt class of a complex linking form is determined by its signature
jumps.  Direct sum therefore adds jumps, while replacing ``t`` by
``exp(2*pi*i*phi) * t^d`` pulls jumps back through the circle map

``theta |-> phi + d*theta  (mod 1)``.

This is exactly the algebra needed by Theorem 4.19.  On its own it is not
always enough to construct the globally normalized twisted signature
function: Yanagida's formula may leave the pairing at ``t=1`` or at
exceptional roots unresolved.  The class
:class:`TwistedSignatureJumpProfile` consequently stores two kinds of
information separately:

* exact, known jump weights; and
* explicit gaps at roots where a local module survives but its pairing has
  not been computed.

Unknown contributions are never replaced by zero.  They remain attached to
their roots under direct sums and are pulled back to every preimage under a
phase/power substitution.

There is one important situation in which the missing root at ``t=1`` can be
recovered without calculating another local pairing.  A representable complex
linking form has total signature jump zero and its averaged signature is
normalized to vanish at ``t=1``.  Therefore, if ``t=1`` is the *only* gap,
its jump is the negative of the sum of all known jumps.  The factory
:func:`averaged_signature_from_representable_profile` performs precisely this
deduction and returns an :class:`AveragedTwistedSignatureFunction`.  It still
refuses profiles with a gap anywhere else on the circle.

The three cases of Theorem 4.19
-------------------------------

Let ``cover_degree`` denote the paper's ``n``, let ``w`` be the winding
number, and put ``h=gcd(n,w)``.

* If ``w=0`` and the representation is ``eta``-regular, only the pattern form
  contributes.
* If ``w`` is nonzero and divisible by ``n``, the companion contribution is a
  direct sum of ``n`` ordinary Blanchfield forms
  ``Bl(K)(phase_i * t^(w/n))``.
* If ``w`` is nonzero and not divisible by ``n``, there are ``h`` twisted
  companion forms of cover degree ``n/h``, each evaluated at
  ``phase_i * t^(w/h)``.

The function :func:`theorem_4_19_signature_jumps` implements exactly these
three branches.  The caller supplies the phase arguments and, in the final
branch, the lower-cover companion profiles.  For standard torus cables in the
GA-knot model, :func:`induced_companion_characters` performs the separate
topological step: it supplies both the lower-cover characters ``chi_i`` and
the values on ``t_Q^(i-1)q_Q(mu_Q^{-w} eta)``.  Keeping that coordinate
calculation outside this algebraic layer makes the assumptions visible and
also allows this function to accept profiles obtained by other methods.

Notation warning
----------------

The symbols in Yanagida's torus-knot paper and in Theorem 4.19 overlap.  For
``YanagidaTorusData(m, n, b)``, ``m`` is the representation/cover degree and
``n`` is the order of the roots containing the character coordinates.  Thus a
Yanagida profile used as the pattern for a cover of degree ``N`` must satisfy
``data.m == N``; comparing ``data.n`` with ``N`` would be incorrect.
"""

from collections import Counter
from dataclasses import dataclass, field

from sage.all import Integer, QQ, gcd

from gaknot.invariants.signature import SignatureFunction
from gaknot.invariants.torus_twisted_signature import (
    YanagidaSignatureProfile,
)
from gaknot.utils.utility import mod_one


def _validated_integer(value, name, *, positive=False, nonzero=False):
    """Return an exact Sage integer after enforcing the requested bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, Integer)):
        raise TypeError(f"{name} must be an integer.")
    value = Integer(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonzero and value == 0:
        raise ValueError(f"{name} must be nonzero.")
    return value


def _validated_jump_weight(value):
    """Return an exact integral jump weight as a Sage ``Integer``.

    ``LT_signature`` can produce the exact Sage rational ``QQ(-1)`` when a
    negative exponent is evaluated, even though its mathematical value is an
    integer.  Accept exact rationals with denominator one while continuing to
    reject nonintegral or floating-point weights.
    """
    if isinstance(value, (bool, float, complex, str)) or value is None:
        raise TypeError("jump weight must be an integer.")
    try:
        rational_value = QQ(value)
    except (TypeError, ValueError):
        raise TypeError("jump weight must be an integer.") from None
    if not rational_value.is_integer():
        raise TypeError("jump weight must be an integer.")
    return Integer(rational_value)


def _validated_argument(value, name="argument"):
    r"""Return an exact rational representative in the interval ``[0,1)``.

    Roots in this module are roots of unity and are encoded by their rational
    arguments.  Floating-point values are rejected even when they happen to
    have a short rational expansion: exact coincidence of independently
    computed roots is required for jump cancellation.
    """
    if isinstance(value, (bool, float, complex, str)) or value is None:
        raise TypeError(f"{name} must be an exact rational number.")
    try:
        value = QQ(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be an exact rational number."
        ) from None
    return QQ(mod_one(value))


def _validated_optional_cover_degree(value):
    """Validate optional representation-degree metadata."""
    if value is None:
        return None
    return _validated_integer(value, "cover_degree", positive=True)


@dataclass(frozen=True)
class SignatureJumpGap:
    r"""One root at which the total jump contains an unknown contribution.

    ``reason`` states the mathematical obstruction, while ``source`` records
    which pattern or companion summand introduced it.  Several gaps may share
    an argument.  They are intentionally retained separately because unknown
    contributions from different summands cannot be cancelled without first
    computing their signs.
    """

    argument: object
    reason: str
    source: str = ""

    def __post_init__(self):
        object.__setattr__(
            self,
            "argument",
            _validated_argument(self.argument),
        )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a nonempty string.")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string.")


@dataclass(frozen=True)
class TwistedSignatureJumpProfile:
    r"""An exact, possibly partial collection of complex signature jumps.

    ``known_jumps`` is normalized to a sorted tuple of ``(argument, weight)``
    pairs.  Duplicate arguments are added and exact cancellations are removed.
    ``unresolved`` lists roots carrying unknown contributions.  Consequently
    :attr:`is_complete` means that every jump of the represented linking form
    is known, not merely that every stored coefficient is nonzero.

    ``cover_degree`` is optional metadata describing the metabelian
    representation.  Theorem 4.19 requires it for pattern and twisted
    companion inputs so that incompatible covers cannot be combined by
    accident.  It is not the number of stored roots or module generators.
    """

    known_jumps: tuple = field(default_factory=tuple)
    unresolved: tuple = field(default_factory=tuple)
    cover_degree: object = None
    label: str = ""

    def __post_init__(self):
        if not isinstance(self.known_jumps, (tuple, list)):
            raise TypeError("known_jumps must be a tuple or list of pairs.")

        # Counter performs the direct-sum addition at coincident roots.  Zero
        # totals are discarded because they carry no Witt-class information.
        counter = Counter()
        for entry in self.known_jumps:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise TypeError(
                    "Every known jump must be an (argument, weight) pair."
                )
            argument = _validated_argument(entry[0])
            weight = _validated_jump_weight(entry[1])
            counter[argument] += weight

        normalized_jumps = tuple(sorted(
            (argument, weight)
            for argument, weight in counter.items()
            if weight != 0
        ))

        if not isinstance(self.unresolved, (tuple, list)):
            raise TypeError("unresolved must be a tuple or list of gaps.")
        normalized_gaps = []
        for gap in self.unresolved:
            if not isinstance(gap, SignatureJumpGap):
                raise TypeError(
                    "unresolved must contain SignatureJumpGap objects."
                )
            normalized_gaps.append(gap)
        normalized_gaps.sort(
            key=lambda gap: (gap.argument, gap.source, gap.reason)
        )

        if not isinstance(self.label, str):
            raise TypeError("label must be a string.")

        object.__setattr__(self, "known_jumps", normalized_jumps)
        object.__setattr__(self, "unresolved", tuple(normalized_gaps))
        object.__setattr__(
            self,
            "cover_degree",
            _validated_optional_cover_degree(self.cover_degree),
        )

    @property
    def is_complete(self):
        """Return whether the profile contains no unresolved local pairing."""
        return not self.unresolved

    @property
    def unresolved_arguments(self):
        """Return the sorted distinct arguments at which a jump is unknown."""
        return tuple(sorted({gap.argument for gap in self.unresolved}))

    @property
    def known_counter(self):
        """Return a defensive ``Counter`` containing the proved contributions."""
        return Counter(dict(self.known_jumps))

    def known_jump_at(self, argument):
        r"""Return the proved part of the jump, ignoring any unknown part."""
        argument = _validated_argument(argument)
        return self.known_counter[argument]

    def jump_at(self, argument):
        r"""Return the total jump or raise when this root has a coverage gap."""
        argument = _validated_argument(argument)
        gaps = tuple(
            gap for gap in self.unresolved if gap.argument == argument
        )
        if gaps:
            sources = tuple(
                gap.source or "unspecified source" for gap in gaps
            )
            raise NotImplementedError(
                f"The jump at argument {argument} has unresolved "
                f"contributions from {sources}."
            )
        return self.known_jump_at(argument)

    def __add__(self, other):
        """Add proved jumps and retain every unresolved direct-sum summand."""
        if not isinstance(other, TwistedSignatureJumpProfile):
            return NotImplemented

        if self.label and other.label:
            label = f"{self.label} + {other.label}"
        else:
            label = self.label or other.label

        # A direct sum does not in general preserve a single representation
        # degree: Theorem 4.19 itself mixes pattern and induced companion
        # summands.  Its result factory restores the satellite cover metadata.
        return TwistedSignatureJumpProfile(
            known_jumps=self.known_jumps + other.known_jumps,
            unresolved=self.unresolved + other.unresolved,
            cover_degree=None,
            label=label,
        )

    def pullback(self, phase_argument, power, *, label=None):
        r"""Substitute ``exp(2*pi*i*phase_argument) * t^power``.

        If a source jump lies at argument ``x`` and ``power=d>0``, its
        preimages are

        ``(x - phase_argument + k)/d,  k=0,...,d-1``.

        All carry the same jump weight.  For ``d<0`` the circle map reverses
        orientation: the preimages are ``(phase_argument-x+k)/|d|`` and every
        jump weight changes sign.  A gap is copied to the same preimages but,
        since it has no known sign, requires no orientation adjustment.
        """
        phase_argument = _validated_argument(
            phase_argument,
            "phase_argument",
        )
        power = _validated_integer(power, "power", nonzero=True)
        degree = abs(power)

        transformed_jumps = []
        for argument, weight in self.known_jumps:
            for branch in range(degree):
                if power > 0:
                    preimage = (argument - phase_argument + branch) / degree
                    transformed_weight = weight
                else:
                    preimage = (phase_argument - argument + branch) / degree
                    transformed_weight = -weight
                transformed_jumps.append((mod_one(preimage), transformed_weight))

        transformed_gaps = []
        for gap in self.unresolved:
            for branch in range(degree):
                if power > 0:
                    preimage = (
                        gap.argument - phase_argument + branch
                    ) / degree
                else:
                    preimage = (
                        phase_argument - gap.argument + branch
                    ) / degree
                transformed_gaps.append(
                    SignatureJumpGap(
                        argument=mod_one(preimage),
                        reason=gap.reason,
                        source=gap.source,
                    )
                )

        if label is None:
            label = self.label
        elif not isinstance(label, str):
            raise TypeError("label must be a string.")

        return TwistedSignatureJumpProfile(
            known_jumps=tuple(transformed_jumps),
            unresolved=tuple(transformed_gaps),
            cover_degree=self.cover_degree,
            label=label,
        )


@dataclass(frozen=True)
class AveragedTwistedSignatureFunction:
    r"""A globally normalized averaged signature of a representable form.

    The completed :attr:`jump_profile` contains every half-jump ``delta`` and
    must satisfy

    ``sum_(xi in S^1) delta(xi) = 0``.

    This is the representability condition from Corollary 5.15 of
    Borodzik--Conway--Politarczyk I.  Property (S-5) in that paper also fixes
    the additive constant of the averaged function by

    ``sigma_av(1) = 0``.

    Write ``delta_1`` for the jump at argument zero, which represents the root
    ``t=1``.  At ``x`` strictly between zero and one the resulting midpoint
    value is

    ``delta_1 + 2*sum_(0<y<x) delta(y) + delta(x)``.

    The one-sided limits differ by ``2*delta(x)``.  At argument zero their
    average is zero, as required, even when ``delta_1`` itself is nonzero.
    This vertical normalization is why the class is separate from the
    classical :class:`SignatureFunction`, whose knots have no jump at one and
    therefore need no independent additive constant.

    ``root_one_jump_inferred`` records whether the factory recovered the
    ``t=1`` jump from representability.  It is diagnostic provenance, not an
    additional mathematical assumption made by this class.
    """

    jump_profile: TwistedSignatureJumpProfile = field(repr=False)
    root_one_jump_inferred: bool = False

    def __post_init__(self):
        if not isinstance(self.jump_profile, TwistedSignatureJumpProfile):
            raise TypeError(
                "jump_profile must be a TwistedSignatureJumpProfile object."
            )
        if not self.jump_profile.is_complete:
            raise ValueError(
                "An averaged twisted signature requires a complete jump profile."
            )
        if not isinstance(self.root_one_jump_inferred, bool):
            raise TypeError("root_one_jump_inferred must be a boolean.")

        total_jump = sum(
            (weight for _, weight in self.jump_profile.known_jumps),
            Integer(0),
        )
        if total_jump != 0:
            raise ValueError(
                "A representable twisted linking form must have total "
                "signature jump zero."
            )

    @property
    def jumps_counter(self):
        """Return a defensive counter containing every completed half-jump."""
        return self.jump_profile.known_counter

    @property
    def total_jump(self):
        """Return the sum of the completed jump weights, necessarily zero."""
        return sum(self.jumps_counter.values(), Integer(0))

    @property
    def root_one_jump(self):
        """Return the completed half-jump at ``t=1`` (argument zero)."""
        return self.jumps_counter[QQ(0)]

    def jump_at(self, argument):
        """Return the completed half-jump at an exact rational argument."""
        return self.jump_profile.jump_at(argument)

    def left_limit(self, argument):
        r"""Return the limit approached counterclockwise from below.

        At argument zero this means the limit near ``1`` from the end of the
        fundamental interval.  Representability and zero total jump make it
        ``-delta_1``.  Away from zero, integration starts with the right-hand
        value ``delta_1`` immediately after crossing ``t=1``.
        """
        argument = _validated_argument(argument)
        delta_one = self.root_one_jump
        if argument == 0:
            return -delta_one
        return delta_one + 2 * sum(
            (
                weight
                for root_argument, weight in self.jump_profile.known_jumps
                if 0 < root_argument < argument
            ),
            Integer(0),
        )

    def right_limit(self, argument):
        """Return the limit immediately after crossing ``argument``."""
        argument = _validated_argument(argument)
        return self.left_limit(argument) + 2 * self.jump_at(argument)

    def __call__(self, argument):
        r"""Evaluate the averaged signature using the midpoint convention."""
        argument = _validated_argument(argument)
        return (
            self.left_limit(argument) + self.right_limit(argument)
        ) / 2


def averaged_signature_from_representable_profile(profile):
    r"""Complete ``t=1`` and normalize a representable twisted signature.

    Args:
        profile: A :class:`TwistedSignatureJumpProfile` known independently to
            arise from a representable complex linking form.  It may already
            be complete, or all of its gaps may lie at argument zero.

    Returns:
        An :class:`AveragedTwistedSignatureFunction`.  If argument zero was
        unresolved, its aggregate jump is inferred as the negative sum of all
        known jumps.  A zero inferred jump is valid and is removed by the
        profile's ordinary sparse normalization.

    Raises:
        TypeError: If ``profile`` has the wrong type.
        NotImplementedError: If any unresolved local contribution occurs away
            from ``t=1``.  Representability determines only the *total* jump
            and cannot separate such missing contributions root by root.
        ValueError: If a profile that already claims completeness has nonzero
            total jump and therefore contradicts the representability
            hypothesis.

    The factory does not attempt to prove representability from raw jump data.
    Its caller must establish that hypothesis, for example from Theorem 4.14
    for a nontrivial prime-power-order Casson--Gordon character.
    """
    if not isinstance(profile, TwistedSignatureJumpProfile):
        raise TypeError(
            "profile must be a TwistedSignatureJumpProfile object."
        )

    nontrivial_gaps = tuple(
        gap for gap in profile.unresolved if gap.argument != 0
    )
    if nontrivial_gaps:
        arguments = tuple(sorted({gap.argument for gap in nontrivial_gaps}))
        raise NotImplementedError(
            "Cannot construct the averaged twisted signature while local "
            f"jumps remain unresolved at nontrivial arguments {arguments}."
        )

    root_one_gaps = tuple(
        gap for gap in profile.unresolved if gap.argument == 0
    )
    if not root_one_gaps:
        return AveragedTwistedSignatureFunction(profile)

    inferred_root_one_jump = -sum(
        (weight for _, weight in profile.known_jumps),
        Integer(0),
    )
    completed_profile = TwistedSignatureJumpProfile(
        known_jumps=profile.known_jumps + (
            (QQ(0), inferred_root_one_jump),
        ),
        unresolved=(),
        cover_degree=profile.cover_degree,
        label=profile.label,
    )
    return AveragedTwistedSignatureFunction(
        completed_profile,
        root_one_jump_inferred=True,
    )


def classical_signature_jump_profile(signature, *, label=""):
    r"""Convert a complete classical ``SignatureFunction`` to jump data."""
    if not isinstance(signature, SignatureFunction):
        raise TypeError("signature must be a SignatureFunction object.")
    return TwistedSignatureJumpProfile(
        known_jumps=tuple(signature.jumps_counter.items()),
        unresolved=(),
        cover_degree=1,
        label=label or signature.plot_title,
    )


def yanagida_twisted_signature_jump_profile(profile, *, label=""):
    r"""Convert Yanagida coverage to a safe global partial jump profile.

    Generic jumps and zero-module exceptional roots are already resolved by
    :class:`YanagidaSignatureProfile`.  Positive-dimensional exceptional roots
    become explicit gaps.  The additional gap at argument zero records that
    Yanagida's Theorems 1.2 and 1.3 both exclude ``t=1``.  It is retained even
    when every nontrivial root is resolved, because equation (18)'s rational
    denominator does not determine the ``(t-1)``-primary Blanchfield pairing.
    """
    if not isinstance(profile, YanagidaSignatureProfile):
        raise TypeError("profile must be a YanagidaSignatureProfile object.")

    gaps = [
        SignatureJumpGap(
            argument=result.argument,
            reason=(
                "Yanagida's Theorem 1.3 excludes this exceptional root, "
                "where Theta detects a positive-dimensional local module."
            ),
            source=f"Yanagida exceptional root a={int(result.a)}",
        )
        for result in profile.unresolved_exceptional_roots
    ]
    gaps.append(
        SignatureJumpGap(
            argument=QQ(0),
            reason=(
                "Yanagida's local module and pairing theorems exclude a=0, "
                "so the jump at t=1 has not been determined."
            ),
            source="Yanagida root a=0",
        )
    )

    return TwistedSignatureJumpProfile(
        known_jumps=profile.known_jump_values,
        unresolved=tuple(gaps),
        cover_degree=profile.data.m,
        label=label or (
            f"Yanagida T({int(profile.data.m)},{int(profile.data.n)})"
        ),
    )


def _coerce_twisted_profile(value, expected_cover_degree, role):
    """Convert supported inputs and verify Theorem 4.19's cover degree."""
    if isinstance(value, YanagidaSignatureProfile):
        value = yanagida_twisted_signature_jump_profile(value)
    elif not isinstance(value, TwistedSignatureJumpProfile):
        raise TypeError(
            f"{role} must be a TwistedSignatureJumpProfile or "
            "YanagidaSignatureProfile object."
        )

    if value.cover_degree is None:
        raise ValueError(
            f"{role} must declare its metabelian cover_degree."
        )
    if value.cover_degree != expected_cover_degree:
        raise ValueError(
            f"{role} has cover_degree={int(value.cover_degree)}, but "
            f"Theorem 4.19 requires {int(expected_cover_degree)}."
        )
    return value


@dataclass(frozen=True)
class Theorem419SignatureResult:
    r"""Structured signature-jump output of Theorem 4.19.

    ``pattern_profile`` is the unchanged pattern contribution.
    ``companion_summands`` contains the already phase/power-pulled-back
    companion contributions, making the decomposition inspectable in tests
    and later diagnostics.  ``total_profile`` is their exact direct sum.
    """

    cover_degree: object
    winding: object
    h: object
    case: str
    phase_arguments: tuple
    pattern_profile: TwistedSignatureJumpProfile = field(repr=False)
    companion_summands: tuple = field(repr=False)
    total_profile: TwistedSignatureJumpProfile = field(repr=False)

    def __post_init__(self):
        cover_degree = _validated_integer(
            self.cover_degree,
            "cover_degree",
            positive=True,
        )
        winding = _validated_integer(self.winding, "winding")
        h = _validated_integer(self.h, "h", positive=True)
        expected_h = Integer(gcd(cover_degree, abs(winding)))
        if h != expected_h:
            raise ValueError("h must equal gcd(cover_degree, winding).")
        if self.case not in {
            "winding_zero",
            "ordinary_companion",
            "metabelian_companion",
        }:
            raise ValueError("case is not a recognized Theorem 4.19 branch.")
        if not isinstance(self.phase_arguments, tuple):
            raise TypeError("phase_arguments must be a tuple.")
        normalized_phases = tuple(
            _validated_argument(value, f"phase_arguments[{index}]")
            for index, value in enumerate(self.phase_arguments)
        )
        if not isinstance(self.pattern_profile, TwistedSignatureJumpProfile):
            raise TypeError("pattern_profile must be a jump profile.")
        if self.pattern_profile.cover_degree != cover_degree:
            raise ValueError(
                "pattern_profile must carry the satellite cover degree."
            )
        if not isinstance(self.companion_summands, tuple) or any(
            not isinstance(item, TwistedSignatureJumpProfile)
            for item in self.companion_summands
        ):
            raise TypeError("companion_summands must be a tuple of profiles.")
        if not isinstance(self.total_profile, TwistedSignatureJumpProfile):
            raise TypeError("total_profile must be a jump profile.")
        if self.total_profile.cover_degree != cover_degree:
            raise ValueError(
                "total_profile must carry the satellite cover degree."
            )

        # The branch label and the number of summands are mathematical data,
        # not display metadata.  Validate them even though the public factory
        # already constructs the expected combinations.
        if winding == 0:
            expected_case = "winding_zero"
            expected_summands = 0
        elif winding % cover_degree == 0:
            expected_case = "ordinary_companion"
            expected_summands = cover_degree
        else:
            expected_case = "metabelian_companion"
            expected_summands = h
        if self.case != expected_case:
            raise ValueError(
                "case does not match cover_degree and winding."
            )
        if len(self.companion_summands) != expected_summands:
            raise ValueError(
                "companion_summands has the wrong length for this branch."
            )
        if len(normalized_phases) != expected_summands:
            raise ValueError(
                "phase_arguments has the wrong length for this branch."
            )

        object.__setattr__(self, "cover_degree", cover_degree)
        object.__setattr__(self, "winding", winding)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "phase_arguments", normalized_phases)

    @property
    def is_complete(self):
        """Return whether every satellite signature jump is determined."""
        return self.total_profile.is_complete

    @property
    def unresolved_arguments(self):
        """Return the roots still requiring local pairing computations."""
        return self.total_profile.unresolved_arguments


def theorem_4_19_signature_jumps(
    pattern_profile,
    cover_degree,
    winding,
    *,
    phase_arguments=(),
    ordinary_companion_signature=None,
    metabelian_companion_profiles=(),
    eta_regular=None,
):
    r"""Assemble signature jumps using Theorem 4.19.

    Args:
        pattern_profile: A partial twisted jump profile, or a Yanagida profile,
            for the pattern representation of cover degree ``cover_degree``.
        cover_degree: The theorem's integer ``n>1``.
        winding: The signed winding number ``w=lk(eta,P)``.
        phase_arguments: Rational arguments of the roots of unity multiplying
            the companion variables.  The divisible branch requires ``n``
            arguments; the nondivisible branch requires ``h=gcd(n,w)``.
        ordinary_companion_signature: The classical ``SignatureFunction`` of
            ``K`` used only when ``n`` divides a nonzero ``w``.
        metabelian_companion_profiles: The ``h`` lower-cover profiles used only
            when ``n`` does not divide a nonzero ``w``.  Each must have cover
            degree ``n/h``.
        eta_regular: For ``w=0``, this must be explicitly ``True`` because the
            theorem applies only under its character-dependent regularity
            condition.  For ``w!=0`` regularity is automatic; passing
            ``False`` is treated as contradictory input.

    Returns:
        An immutable :class:`Theorem419SignatureResult` exposing the theorem
        branch, transformed summands, exact known jumps and all propagated
        coverage gaps.

    This function implements the algebra after the induced characters and
    phase exponents have been determined.  It does not derive those inputs
    from ``H_1`` or from a generalized algebraic-knot description.
    """
    cover_degree = _validated_integer(
        cover_degree,
        "cover_degree",
        positive=True,
    )
    if cover_degree <= 1:
        raise ValueError("cover_degree must be greater than one.")
    winding = _validated_integer(winding, "winding")

    if eta_regular is not None and not isinstance(eta_regular, bool):
        raise TypeError("eta_regular must be True, False or None.")
    if not isinstance(phase_arguments, (tuple, list)):
        raise TypeError("phase_arguments must be a tuple or list.")
    phase_arguments = tuple(
        _validated_argument(value, f"phase_arguments[{index}]")
        for index, value in enumerate(phase_arguments)
    )
    if not isinstance(metabelian_companion_profiles, (tuple, list)):
        raise TypeError(
            "metabelian_companion_profiles must be a tuple or list."
        )
    metabelian_companion_profiles = tuple(metabelian_companion_profiles)

    pattern = _coerce_twisted_profile(
        pattern_profile,
        cover_degree,
        "pattern_profile",
    )
    h = Integer(gcd(cover_degree, abs(winding)))

    if winding == 0:
        # The nonzero-winding part of Lemma 4.22 gives regularity for free.
        # At winding zero the theorem instead imposes a condition on every
        # induced character, which this algebraic layer cannot reconstruct.
        if eta_regular is not True:
            raise ValueError(
                "eta_regular=True is required for the winding-zero branch."
            )
        if phase_arguments:
            raise ValueError(
                "phase_arguments are not used when winding is zero."
            )
        if ordinary_companion_signature is not None:
            raise ValueError(
                "The companion does not contribute when winding is zero."
            )
        if metabelian_companion_profiles:
            raise ValueError(
                "The companion does not contribute when winding is zero."
            )

        total = TwistedSignatureJumpProfile(
            known_jumps=pattern.known_jumps,
            unresolved=pattern.unresolved,
            cover_degree=cover_degree,
            label=pattern.label,
        )
        return Theorem419SignatureResult(
            cover_degree=cover_degree,
            winding=winding,
            h=h,
            case="winding_zero",
            phase_arguments=(),
            pattern_profile=pattern,
            companion_summands=(),
            total_profile=total,
        )

    if eta_regular is False:
        raise ValueError(
            "Theorem 4.19 guarantees eta-regularity when winding is nonzero."
        )

    companion_summands = []
    if winding % cover_degree == 0:
        # In this branch the restriction to each companion component is
        # one-dimensional and hence uses the ordinary Blanchfield pairing.
        if not isinstance(ordinary_companion_signature, SignatureFunction):
            raise TypeError(
                "ordinary_companion_signature must be supplied as a "
                "SignatureFunction when cover_degree divides winding."
            )
        if metabelian_companion_profiles:
            raise ValueError(
                "Twisted companion profiles are not used when cover_degree "
                "divides winding."
            )
        if len(phase_arguments) != cover_degree:
            raise ValueError(
                "The divisible branch requires one phase argument for each "
                f"of the {int(cover_degree)} companion summands."
            )

        exponent = winding // cover_degree
        classical = classical_signature_jump_profile(
            ordinary_companion_signature,
            label="ordinary companion",
        )
        for index, phase in enumerate(phase_arguments):
            companion_summands.append(
                classical.pullback(
                    phase,
                    exponent,
                    label=f"ordinary companion summand {index + 1}",
                )
            )
        case = "ordinary_companion"
    else:
        # The companion cover drops from n to n/h.  There are h induced
        # characters and therefore h potentially different twisted profiles.
        if ordinary_companion_signature is not None:
            raise ValueError(
                "ordinary_companion_signature is used only when "
                "cover_degree divides winding."
            )
        if len(phase_arguments) != h:
            raise ValueError(
                "The nondivisible branch requires one phase argument for "
                f"each of the h={int(h)} companion summands."
            )
        if len(metabelian_companion_profiles) != h:
            raise ValueError(
                "The nondivisible branch requires exactly "
                f"h={int(h)} twisted companion profiles."
            )

        lower_cover_degree = cover_degree // h
        exponent = winding // h
        for index, (companion, phase) in enumerate(zip(
            metabelian_companion_profiles,
            phase_arguments,
        )):
            companion = _coerce_twisted_profile(
                companion,
                lower_cover_degree,
                f"metabelian_companion_profiles[{index}]",
            )
            companion_summands.append(
                companion.pullback(
                    phase,
                    exponent,
                    label=f"metabelian companion summand {index + 1}",
                )
            )
        case = "metabelian_companion"

    combined = pattern
    for summand in companion_summands:
        combined = combined + summand
    total = TwistedSignatureJumpProfile(
        known_jumps=combined.known_jumps,
        unresolved=combined.unresolved,
        cover_degree=cover_degree,
        label=combined.label,
    )

    return Theorem419SignatureResult(
        cover_degree=cover_degree,
        winding=winding,
        h=h,
        case=case,
        phase_arguments=phase_arguments,
        pattern_profile=pattern,
        companion_summands=tuple(companion_summands),
        total_profile=total,
    )
