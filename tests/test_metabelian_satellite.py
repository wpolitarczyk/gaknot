r"""Tests for the signature-jump form of Theorem 4.19.

The theorem in *Twisted Blanchfield pairings and twisted signatures II*
decomposes a satellite's metabelian Blanchfield form into a pattern summand and
phase/power substitutions of companion summands.  This suite tests the two
logically separate layers used to encode that statement:

1. ``TwistedSignatureJumpProfile`` implements the Witt-class algebra.  Direct
   sum adds jumps, and substitution by ``exp(2*pi*i*phi)t^d`` pulls a jump back
   through ``theta -> phi+d*theta``.  Coverage gaps must undergo the same root
   transformations without being assigned guessed weights.
2. ``theorem_4_19_signature_jumps`` selects the correct decomposition for
   ``w=0``, ``n|w`` and ``n`` not dividing ``w``.  It checks the number of
   phase arguments, the companion type and every metabelian cover degree.

Most examples use deliberately tiny synthetic profiles.  Their expected
preimages can be calculated directly from rational arithmetic, independently
of the implementation.  Separate integration tests convert Yanagida's actual
torus-knot output and check that its unresolved exceptional roots and the
unsupported root ``t=1`` remain visible.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import QQ

from gaknot import (
    AveragedTwistedSignatureFunction,
    SignatureFunction,
    SignatureJumpGap,
    Theorem419SignatureResult,
    TwistedSignatureJumpProfile,
    YanagidaTorusData,
    averaged_signature_from_representable_profile,
    classical_signature_jump_profile,
    theorem_4_19_signature_jumps,
    yanagida_signature_profile,
    yanagida_twisted_signature_jump_profile,
)


# ---------------------------------------------------------------------------
# Exact partial-profile algebra
# ---------------------------------------------------------------------------

def test_jump_profile_normalizes_roots_and_aggregates_exact_cancellations():
    r"""Equivalent rational arguments represent the same point of the circle.

    The first two entries below both lie at ``1/3`` modulo one and therefore
    add to three.  The two entries at ``2/3`` cancel exactly.  The gap at
    ``5/4`` is normalized independently to ``1/4``.  This is the basic
    canonicalization on which later direct-sum cancellations depend.
    """
    gap = SignatureJumpGap(
        QQ(5) / 4,
        reason="the local pairing has not been computed",
        source="example",
    )

    profile = TwistedSignatureJumpProfile(
        known_jumps=(
            (QQ(1) / 3, 1),
            (QQ(4) / 3, 2),
            (QQ(2) / 3, -1),
            (QQ(2) / 3, 1),
        ),
        unresolved=(gap,),
        cover_degree=3,
        label="normalization example",
    )

    assert profile.known_jumps == ((QQ(1) / 3, 3),)
    assert profile.unresolved_arguments == (QQ(1) / 4,)
    assert profile.cover_degree == 3
    assert not profile.is_complete

    # ``known_jump_at`` reports the proved part even at an ordinary point,
    # whereas ``jump_at`` protects callers from treating a gap as total zero.
    assert profile.known_jump_at(QQ(4) / 3) == 3
    assert profile.known_jump_at(QQ(5) / 4) == 0
    with pytest.raises(NotImplementedError, match="argument 1/4"):
        profile.jump_at(QQ(1) / 4)


def test_jump_profile_returns_a_defensive_counter():
    r"""Mutating a diagnostic counter cannot alter the immutable profile."""
    profile = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 5, -1),),
        cover_degree=2,
    )

    external_counter = profile.known_counter
    external_counter[QQ(1) / 5] = 99

    assert profile.known_jumps == ((QQ(1) / 5, -1),)
    assert profile.jump_at(QQ(1) / 5) == -1


def test_jump_profile_accepts_exact_integral_sage_rational_weights():
    """Treat ``QQ(-1)`` as the integer weight it represents exactly.

    The elementary Levine--Tristram formula can return a Sage ``Rational``
    with denominator one when ``(-1)`` is obtained from a negative exponent.
    The metabelian profile should normalize that harmless representation detail
    rather than reject a mathematically integral classical signature jump.
    """
    profile = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 6, QQ(-1)),),
        cover_degree=2,
    )

    assert profile.known_jumps == ((QQ(1) / 6, -1),)


def test_direct_sum_cancels_known_terms_but_never_unknown_terms():
    r"""Two unknown contributions cannot be cancelled before computing them.

    The proved jumps ``+1`` and ``-1`` at ``1/4`` disappear in the direct
    sum.  Both summands also contain a gap at ``1/2``.  Even though their
    eventual values might cancel, no theorem currently proves that they do,
    so the result must retain two distinct gaps and reject total lookup there.
    """
    first = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 4, 1),),
        unresolved=(SignatureJumpGap(
            QQ(1) / 2,
            reason="first missing pairing",
            source="first summand",
        ),),
        cover_degree=2,
    )
    second = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 4, -1), (QQ(1) / 3, 2)),
        unresolved=(SignatureJumpGap(
            QQ(1) / 2,
            reason="second missing pairing",
            source="second summand",
        ),),
        cover_degree=2,
    )

    total = first + second

    assert total.known_jumps == ((QQ(1) / 3, 2),)
    assert len(total.unresolved) == 2
    assert total.unresolved_arguments == (QQ(1) / 2,)
    # A generic direct sum does not claim one metabelian cover degree; the
    # structured Theorem 4.19 constructor restores that metadata on its total.
    assert total.cover_degree is None
    with pytest.raises(NotImplementedError, match="first summand"):
        total.jump_at(QQ(1) / 2)


def test_positive_phase_power_pullback_transforms_jumps_and_gaps():
    r"""Solve ``phase + 2*theta = source`` for every source root.

    With phase ``1/6``, the known source root ``1/3`` has preimages

    ``(1/3-1/6)/2 = 1/12`` and ``(1/3-1/6+1)/2 = 7/12``.

    The unresolved source root ``1/2`` similarly has preimages ``1/6`` and
    ``2/3``.  The jump weight is copied to both known preimages, while the gap
    reason and provenance are copied to both unresolved preimages.
    """
    profile = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 3, 2),),
        unresolved=(SignatureJumpGap(
            QQ(1) / 2,
            reason="missing residue form",
            source="companion",
        ),),
        cover_degree=3,
    )

    pulled_back = profile.pullback(QQ(1) / 6, 2)

    assert pulled_back.known_jumps == (
        (QQ(1) / 12, 2),
        (QQ(7) / 12, 2),
    )
    assert pulled_back.unresolved_arguments == (
        QQ(1) / 6,
        QQ(2) / 3,
    )
    assert all(gap.source == "companion" for gap in pulled_back.unresolved)
    assert pulled_back.cover_degree == 3


def test_negative_power_reverses_the_orientation_of_signature_jumps():
    r"""A negative winding reverses each one-sided jump difference.

    For phase ``1/6`` and power ``-2``, the source root ``1/3`` pulls back to
    ``11/12`` and ``5/12``.  Traversing the target circle positively traverses
    the source circle negatively, so the source weight ``+1`` becomes ``-1``
    at both roots.  Coverage gaps move to their preimages but have no sign to
    reverse.
    """
    profile = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 3, 1),),
        unresolved=(SignatureJumpGap(
            QQ(1) / 2,
            reason="unknown sign",
            source="negative-winding companion",
        ),),
        cover_degree=2,
    )

    pulled_back = profile.pullback(QQ(1) / 6, -2)

    assert pulled_back.known_jumps == (
        (QQ(5) / 12, -1),
        (QQ(11) / 12, -1),
    )
    assert pulled_back.unresolved_arguments == (
        QQ(1) / 3,
        QQ(5) / 6,
    )


@pytest.mark.parametrize("bad_argument", [0.25, "1/4", None, True])
def test_jump_profiles_reject_inexact_or_ambiguous_arguments(bad_argument):
    r"""Root matching must never depend on floating-point approximations."""
    with pytest.raises(TypeError, match="exact rational"):
        TwistedSignatureJumpProfile(known_jumps=((bad_argument, 1),))


# ---------------------------------------------------------------------------
# Representability and globally normalized averaged signatures
# ---------------------------------------------------------------------------

def test_representability_completes_root_one_and_fixes_vertical_normalization():
    r"""Recover the only missing jump and integrate it around the circle.

    The synthetic profile has known half-jumps ``+1`` at ``1/4`` and ``-2``
    at ``3/4``.  Their sum is ``-1``.  For a representable complex linking
    form, the total sum of all half-jumps must be zero, so a gap confined to
    ``t=1`` is forced to have weight ``+1``.

    The averaged signature is then normalized by ``sigma_av(1)=0``.  Just to
    the right of argument zero its value is ``+1``.  Crossing ``1/4`` changes
    it by ``2*(+1)=2``, giving the regular value ``3``.  Crossing ``3/4``
    changes it by ``2*(-2)=-4``, returning to ``-1`` before the final crossing
    at one.  At either discontinuity the value is the average of those
    one-sided limits.

    This example deliberately uses a nonzero jump at one.  It would expose an
    implementation that merely wrapped the data in the classical
    ``SignatureFunction``, because that class does not carry the independent
    vertical offset needed to make the averaged value at one equal zero.
    """
    partial = TwistedSignatureJumpProfile(
        known_jumps=(
            (QQ(1) / 4, 1),
            (QQ(3) / 4, -2),
        ),
        unresolved=(SignatureJumpGap(
            0,
            reason="the local t=1 pairing is unavailable",
            source="root-one test profile",
        ),),
        cover_degree=2,
        label="representable example",
    )

    signature = averaged_signature_from_representable_profile(partial)

    assert isinstance(signature, AveragedTwistedSignatureFunction)
    assert signature.root_one_jump_inferred
    assert signature.root_one_jump == 1
    assert signature.total_jump == 0
    assert signature.jump_profile.is_complete
    assert signature.jump_profile.known_jumps == (
        (QQ(0), 1),
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -2),
    )

    # The root-one midpoint is the representable normalization.  Its two
    # limits retain the nonzero local jump instead of being flattened to zero.
    assert signature.left_limit(0) == -1
    assert signature(0) == 0
    assert signature.right_limit(0) == 1
    assert signature(1) == 0

    # Verify regular values and midpoint values on both arcs.  The evaluation
    # arguments are exact rationals so there is no approximate root matching.
    assert signature(QQ(1) / 8) == 1
    assert signature.left_limit(QQ(1) / 4) == 1
    assert signature(QQ(1) / 4) == 2
    assert signature.right_limit(QQ(1) / 4) == 3
    assert signature(QQ(1) / 2) == 3
    assert signature.left_limit(QQ(3) / 4) == 3
    assert signature(QQ(3) / 4) == 1
    assert signature.right_limit(QQ(3) / 4) == -1
    assert signature(QQ(7) / 8) == -1


def test_complete_representable_profile_is_preserved_without_inference():
    r"""Do not claim inference when every jump was supplied explicitly."""
    complete = TwistedSignatureJumpProfile(
        known_jumps=((QQ(0), 1), (QQ(1) / 3, -1)),
        cover_degree=3,
    )

    signature = averaged_signature_from_representable_profile(complete)

    assert signature.jump_profile is complete
    assert not signature.root_one_jump_inferred
    assert signature.total_jump == 0


def test_representability_does_not_resolve_a_gap_away_from_root_one():
    r"""A zero-total constraint cannot locate an unknown nontrivial jump.

    Even if representability determines the sum of all missing contributions,
    it does not determine the individual contribution at ``1/3``.  Therefore
    the factory must preserve the package's coverage discipline and stop.
    """
    partial = TwistedSignatureJumpProfile(
        unresolved=(
            SignatureJumpGap(
                0,
                reason="unknown root-one contribution",
                source="root one",
            ),
            SignatureJumpGap(
                QQ(1) / 3,
                reason="unknown exceptional contribution",
                source="exceptional root",
            ),
        ),
        cover_degree=3,
    )

    with pytest.raises(NotImplementedError, match="nontrivial arguments.*1/3"):
        averaged_signature_from_representable_profile(partial)


def test_complete_nonzero_total_jump_contradicts_representability():
    r"""Reject complete data that violate the theorem used for normalization."""
    inconsistent = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 4, 1),),
        cover_degree=2,
    )

    with pytest.raises(ValueError, match="total signature jump zero"):
        averaged_signature_from_representable_profile(inconsistent)


@pytest.mark.parametrize("bad_argument", [0.5, "1/2", None, True])
def test_averaged_signature_evaluation_requires_exact_arguments(bad_argument):
    r"""Evaluation uses the same exact-root contract as the jump profiles."""
    signature = averaged_signature_from_representable_profile(
        TwistedSignatureJumpProfile(cover_degree=2)
    )

    with pytest.raises(TypeError, match="exact rational"):
        signature(bad_argument)


# ---------------------------------------------------------------------------
# Conversion from existing signature objects
# ---------------------------------------------------------------------------

def test_classical_signature_conversion_is_complete_and_exact():
    r"""An ordinary ``SignatureFunction`` already contains all of its jumps."""
    signature = SignatureFunction(values=[
        (QQ(1) / 6, -1),
        (QQ(5) / 6, 1),
    ])

    profile = classical_signature_jump_profile(
        signature,
        label="trefoil",
    )

    assert profile.known_jumps == (
        (QQ(1) / 6, -1),
        (QQ(5) / 6, 1),
    )
    assert profile.unresolved == ()
    assert profile.is_complete
    assert profile.cover_degree == 1
    assert profile.label == "trefoil"


def test_yanagida_conversion_adds_only_the_genuine_missing_roots_for_t_two():
    r"""Zero-module exceptional roots are resolved, while ``t=1`` is not.

    For ``T(2,5), b=(1,4)``, Yanagida's generic formula computes jumps ``-1``
    and ``+1`` at ``2/5`` and ``3/5``.  The exceptional roots ``1/5`` and
    ``4/5`` have zero local module and need no gaps.  The papers still provide
    no local formula at ``a=0``, so conversion adds exactly one gap at zero.
    """
    yanagida = yanagida_signature_profile(
        YanagidaTorusData(2, 5, (1, 4))
    )

    profile = yanagida_twisted_signature_jump_profile(yanagida)

    assert profile.known_jumps == (
        (QQ(2) / 5, -1),
        (QQ(3) / 5, 1),
    )
    assert profile.unresolved_arguments == (QQ(0),)
    assert profile.cover_degree == 2
    assert not profile.is_complete

    # Roots with a certified zero module remain ordinary resolved points.
    assert profile.jump_at(QQ(1) / 5) == 0
    assert profile.jump_at(QQ(4) / 5) == 0
    with pytest.raises(NotImplementedError, match="Yanagida root a=0"):
        profile.jump_at(0)


def test_yanagida_conversion_preserves_surviving_exceptional_gaps():
    r"""The global profile exposes every limitation of the local theorem.

    In the ``T(3,4), b=(0,1,3)`` example, ``a=1`` and ``a=3`` each carry a
    one-dimensional exceptional module.  Together with the unsupported root
    ``a=0``, the converted profile must therefore have gaps at ``0,1/4,3/4``.
    The computed generic jump at ``a=2`` is zero and needs no gap.
    """
    yanagida = yanagida_signature_profile(
        YanagidaTorusData(3, 4, (0, 1, 3))
    )

    profile = yanagida_twisted_signature_jump_profile(yanagida)

    assert profile.known_jumps == ()
    assert profile.unresolved_arguments == (
        QQ(0),
        QQ(1) / 4,
        QQ(3) / 4,
    )
    assert profile.cover_degree == 3


# ---------------------------------------------------------------------------
# The three branches of Theorem 4.19
# ---------------------------------------------------------------------------

def test_theorem_419_winding_zero_returns_only_the_pattern():
    r"""The companion disappears exactly in the eta-regular ``w=0`` case.

    A known jump and a coverage gap are included in the synthetic pattern so
    that the test checks preservation of the entire partial profile, not just
    equality of two empty objects.  Since ``gcd(n,0)=n``, the structural result
    records ``h=3`` even though no companion summands are constructed.
    """
    pattern = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 4, 2),),
        unresolved=(SignatureJumpGap(
            0,
            reason="pattern root one is unknown",
            source="pattern",
        ),),
        cover_degree=3,
        label="pattern",
    )

    result = theorem_4_19_signature_jumps(
        pattern,
        3,
        0,
        eta_regular=True,
    )

    assert isinstance(result, Theorem419SignatureResult)
    assert result.case == "winding_zero"
    assert result.cover_degree == 3
    assert result.winding == 0
    assert result.h == 3
    assert result.phase_arguments == ()
    assert result.companion_summands == ()
    assert result.total_profile.known_jumps == pattern.known_jumps
    assert result.total_profile.unresolved == pattern.unresolved
    assert result.total_profile.cover_degree == 3
    assert not result.is_complete


def test_theorem_419_winding_zero_requires_explicit_eta_regularity():
    r"""The character-dependent hypothesis must not be assumed by default."""
    pattern = TwistedSignatureJumpProfile(cover_degree=2)

    with pytest.raises(ValueError, match="eta_regular=True"):
        theorem_4_19_signature_jumps(pattern, 2, 0)
    with pytest.raises(ValueError, match="eta_regular=True"):
        theorem_4_19_signature_jumps(
            pattern,
            2,
            0,
            eta_regular=False,
        )


def test_theorem_419_divisible_winding_uses_ordinary_companion_forms():
    r"""Check all four preimages from two differently phased summands.

    Take cover degree ``n=2`` and winding ``w=4``.  The exponent in Theorem
    4.19 is therefore ``w/n=2`` and there are two ordinary companion terms.
    The companion has jumps ``+1`` at ``1/3`` and ``-1`` at ``2/3``.

    * Phase zero produces ``+1`` at ``1/6,2/3`` and ``-1`` at ``1/3,5/6``.
    * Phase ``1/4`` produces ``+1`` at ``1/24,13/24`` and ``-1`` at
      ``5/24,17/24``.

    An empty complete pattern isolates precisely these transformed companion
    terms and makes the expected counter independent of Yanagida's formulas.
    """
    pattern = TwistedSignatureJumpProfile(
        cover_degree=2,
        label="empty pattern",
    )
    companion = SignatureFunction(values=[
        (QQ(1) / 3, 1),
        (QQ(2) / 3, -1),
    ])

    result = theorem_4_19_signature_jumps(
        pattern,
        2,
        4,
        phase_arguments=(0, QQ(1) / 4),
        ordinary_companion_signature=companion,
    )

    assert result.case == "ordinary_companion"
    assert result.h == 2
    assert len(result.companion_summands) == 2
    assert result.total_profile.known_jumps == (
        (QQ(1) / 24, 1),
        (QQ(1) / 6, 1),
        (QQ(5) / 24, -1),
        (QQ(1) / 3, -1),
        (QQ(13) / 24, 1),
        (QQ(2) / 3, 1),
        (QQ(17) / 24, -1),
        (QQ(5) / 6, -1),
    )
    assert result.total_profile.cover_degree == 2
    assert result.is_complete


def test_theorem_419_nondivisible_winding_uses_lower_cover_profiles():
    r"""Check ``h`` twisted companions and propagate both companion gaps.

    For cover degree ``n=4`` and winding ``w=2``, we have ``h=2``.  The two
    companion inputs must therefore have cover degree ``n/h=2``, and the
    substitution exponent is ``w/h=1``.

    The first phase is zero, so its known root ``1/4`` and gap ``1/2`` stay in
    place.  The second phase is ``1/6``: its known root ``1/3`` moves to
    ``1/6``, while its gap at zero moves to ``5/6``.  The pattern jump at
    ``3/4`` is included unchanged.
    """
    pattern = TwistedSignatureJumpProfile(
        known_jumps=((QQ(3) / 4, 2),),
        cover_degree=4,
        label="pattern",
    )
    first_companion = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 4, 1),),
        unresolved=(SignatureJumpGap(
            QQ(1) / 2,
            reason="first lower-cover pairing is unknown",
            source="first companion",
        ),),
        cover_degree=2,
    )
    second_companion = TwistedSignatureJumpProfile(
        known_jumps=((QQ(1) / 3, -1),),
        unresolved=(SignatureJumpGap(
            0,
            reason="second lower-cover root one is unknown",
            source="second companion",
        ),),
        cover_degree=2,
    )

    result = theorem_4_19_signature_jumps(
        pattern,
        4,
        2,
        phase_arguments=(0, QQ(1) / 6),
        metabelian_companion_profiles=(
            first_companion,
            second_companion,
        ),
    )

    assert result.case == "metabelian_companion"
    assert result.h == 2
    assert len(result.companion_summands) == 2
    assert result.total_profile.known_jumps == (
        (QQ(1) / 6, -1),
        (QQ(1) / 4, 1),
        (QQ(3) / 4, 2),
    )
    assert result.unresolved_arguments == (
        QQ(1) / 2,
        QQ(5) / 6,
    )
    assert result.total_profile.cover_degree == 4
    assert not result.is_complete

    # The known part at an unresolved root is still available diagnostically,
    # but requesting the total must identify its companion provenance.
    assert result.total_profile.known_jump_at(QQ(1) / 2) == 0
    with pytest.raises(NotImplementedError, match="first companion"):
        result.total_profile.jump_at(QQ(1) / 2)


def test_theorem_419_accepts_yanagida_pattern_with_matching_matrix_size():
    r"""Yanagida's ``m``, rather than its ``n``, is the cover degree.

    ``YanagidaTorusData(2,5,...)`` describes a two-dimensional metabelian
    representation whose character coordinates lie in ``Z/5``.  Theorem 4.19
    must therefore accept it for cover degree two.  With winding zero, the
    converted pattern retains its known jumps and its explicit ``t=1`` gap.
    """
    pattern = yanagida_signature_profile(
        YanagidaTorusData(2, 5, (1, 4))
    )

    result = theorem_4_19_signature_jumps(
        pattern,
        2,
        0,
        eta_regular=True,
    )

    assert result.pattern_profile.cover_degree == 2
    assert result.total_profile.known_jumps == (
        (QQ(2) / 5, -1),
        (QQ(3) / 5, 1),
    )
    assert result.unresolved_arguments == (QQ(0),)


def test_theorem_419_rejects_yanagida_cover_degree_not_character_order():
    r"""Guard against confusing the overlapping ``m,n`` notation.

    The same ``T(2,5)`` profile cannot be used as a five-fold pattern profile:
    five is the order of Yanagida's character roots, while the representation
    and cover degree are two.  Accepting this input would select the wrong
    branch and the wrong number of induced companion characters.
    """
    pattern = yanagida_signature_profile(
        YanagidaTorusData(2, 5, (1, 4))
    )

    with pytest.raises(ValueError, match=r"cover_degree=2.*requires 5"):
        theorem_4_19_signature_jumps(
            pattern,
            5,
            0,
            eta_regular=True,
        )


def test_theorem_419_rejects_wrong_lower_companion_cover_degree():
    r"""The nondivisible branch requires companion degree exactly ``n/h``."""
    pattern = TwistedSignatureJumpProfile(cover_degree=4)
    correct = TwistedSignatureJumpProfile(cover_degree=2)
    wrong = TwistedSignatureJumpProfile(cover_degree=3)

    with pytest.raises(ValueError, match=r"cover_degree=3.*requires 2"):
        theorem_4_19_signature_jumps(
            pattern,
            4,
            2,
            phase_arguments=(0, 0),
            metabelian_companion_profiles=(correct, wrong),
        )


def test_theorem_419_validates_the_number_of_phase_arguments_in_each_branch():
    r"""Each phase corresponds to one direct-sum term in the theorem."""
    pattern_two = TwistedSignatureJumpProfile(cover_degree=2)
    classical = SignatureFunction()
    with pytest.raises(ValueError, match="2 companion summands"):
        theorem_4_19_signature_jumps(
            pattern_two,
            2,
            4,
            phase_arguments=(0,),
            ordinary_companion_signature=classical,
        )

    pattern_four = TwistedSignatureJumpProfile(cover_degree=4)
    lower = TwistedSignatureJumpProfile(cover_degree=2)
    with pytest.raises(ValueError, match="h=2"):
        theorem_4_19_signature_jumps(
            pattern_four,
            4,
            2,
            phase_arguments=(0,),
            metabelian_companion_profiles=(lower, lower),
        )


def test_theorem_419_result_and_nested_profiles_are_immutable():
    r"""A checked decomposition cannot later be changed behind its metadata."""
    pattern = TwistedSignatureJumpProfile(cover_degree=2)
    result = theorem_4_19_signature_jumps(
        pattern,
        2,
        0,
        eta_regular=True,
    )

    with pytest.raises(FrozenInstanceError):
        result.case = "ordinary_companion"
    with pytest.raises(FrozenInstanceError):
        result.total_profile.cover_degree = 3
