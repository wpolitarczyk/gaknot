#!/usr/bin/env sage -python

r"""Tests for signature jumps extracted from Yanagida's local pairings.

The matrix tests in ``test_torus_twisted_blanchfield.py`` establish that the
implemented ``Theta`` and ``Psi`` satisfy Yanagida's algebraic identities.
This file tests the logically separate passage from those matrices to Hodge
signs and signature jumps:

1. specialize ``Theta`` at a generic root and describe its cokernel;
2. divide the pairing numerator by the local order;
3. take the coefficient of the resulting simple pole;
4. remove the phase of a standard ``xi``-positive polynomial;
5. calculate the exact inertia of the resulting Hermitian matrix.

The strongest regression examples come from Theorem 1.3 of
Borodzik--Conway--Politarczyk, *Twisted Blanchfield pairings and twisted
signatures III: Applications*.  That theorem gives the elementary-form
decomposition for metabelian pairings of ``T(2,q)``.  It therefore determines
not merely the locations but also the signs of the jumps independently of the
residue algorithm implemented here.

The Applications paper uses ``xi=exp(-2*pi*i/q)`` in the explicit
decomposition, while ``YanagidaTorusData`` uses the cyclotomic generator
``zeta=exp(2*pi*i/q)``.  Thus a summand at ``xi^e`` occurs at the parameter
``a=-e mod q`` in these tests.  Recording this conversion explicitly prevents
an accidental global reversal of all roots or signs.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import I, Integer, QQ, QQbar, identity_matrix, matrix

from gaknot import (
    HermitianInertia,
    SignatureFunction,
    YanagidaExceptionalRoot,
    YanagidaSignatureJump,
    YanagidaSignatureProfile,
    YanagidaTorusData,
    exact_hermitian_inertia,
    yanagida_generic_signature_jumps,
    yanagida_local_signature_jump,
    yanagida_signature_profile,
)


@pytest.fixture(scope="module")
def theta_one_q_five_data():
    r"""The ``T(2,5)`` orbit corresponding to the first character parameter."""
    return YanagidaTorusData(2, 5, (1, 4))


@pytest.fixture(scope="module")
def theta_one_q_seven_data():
    r"""The ``T(2,7)`` orbit ``(theta,-theta)`` for ``theta=1``."""
    return YanagidaTorusData(2, 7, (1, 6))


@pytest.fixture(scope="module")
def theta_two_q_seven_data():
    r"""The ``T(2,7)`` orbit ``(theta,-theta)`` for ``theta=2``."""
    return YanagidaTorusData(2, 7, (2, 5))


def test_hermitian_inertia_of_a_diagonal_form():
    r"""Positive, negative and radical directions are counted separately."""
    hermitian = matrix(QQbar, [[3, 0, 0], [0, -2, 0], [0, 0, 0]])

    inertia = exact_hermitian_inertia(hermitian)

    assert inertia == HermitianInertia(1, 1, 1)
    assert inertia.signature == 0
    assert inertia.dimension == 3


def test_hermitian_inertia_handles_a_complex_hyperbolic_block():
    r"""A zero diagonal must not be confused with a two-dimensional radical.

    The matrix below has eigenvalues of opposite sign.  It exercises the
    branch that first replaces a basis vector to create a nonzero diagonal
    pivot from the complex off-diagonal entry.
    """
    z = QQbar(1 + I)
    hyperbolic = matrix(QQbar, [[0, z], [z.conjugate(), 0]])

    inertia = exact_hermitian_inertia(hyperbolic)

    assert inertia == HermitianInertia(1, 1, 0)


def test_hermitian_inertia_is_invariant_under_exact_congruence():
    r"""Changing the quotient basis must not change the Hodge signs."""
    original = matrix(QQbar, [[2, 0], [0, -3]])
    basis_change = matrix(QQbar, [[1, 1 + I], [2 - I, 1]])
    assert basis_change.det() != 0
    congruent = basis_change.conjugate_transpose() * original * basis_change

    assert exact_hermitian_inertia(congruent) == HermitianInertia(1, 1, 0)


def test_hermitian_inertia_accepts_the_empty_form():
    r"""The zero-dimensional form has zero positive, negative and null index."""
    empty = matrix(QQbar, 0, 0)

    assert exact_hermitian_inertia(empty) == HermitianInertia(0, 0, 0)


def test_hermitian_inertia_rejects_a_nonsquare_matrix():
    """Inertia is defined only for endomorphism matrices."""
    with pytest.raises(ValueError, match="must be square"):
        exact_hermitian_inertia(matrix(QQbar, 2, 3))


def test_hermitian_inertia_rejects_a_nonhermitian_matrix():
    """Silently symmetrizing an input would change the mathematical form."""
    nonhermitian = matrix(QQbar, [[1, 1], [0, 1]])

    with pytest.raises(ValueError, match="must be Hermitian"):
        exact_hermitian_inertia(nonhermitian)


@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_hermitian_inertia_data_rejects_nonintegral_indices(bad_value):
    """Inertia indices are exact dimensions, not coercible numeric labels."""
    with pytest.raises(TypeError, match="must be an integer"):
        HermitianInertia(bad_value, 0, 0)


@pytest.mark.parametrize("indices", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_hermitian_inertia_data_rejects_negative_indices(indices):
    """No positive, negative or zero subspace can have negative dimension."""
    with pytest.raises(ValueError, match="must be nonnegative"):
        HermitianInertia(*indices)


def test_t_two_five_reference_jumps_match_the_explicit_decomposition(
    theta_one_q_five_data,
):
    r"""Recover the two generic Hodge signs for ``T(2,5)``, ``theta=1``.

    The explicit decomposition contains

    ``e(1,+1,xi^2) + e(1,-1,xi^-2)``.

    Since ``xi=zeta^-1``, these occur at ``a=3`` with sign ``+1`` and at
    ``a=2`` with sign ``-1``.  The roots ``a=1,4`` are exceptional and are
    outside Yanagida's pairing theorem.
    """
    results = yanagida_generic_signature_jumps(theta_one_q_five_data)

    assert {result.a: result.jump for result in results} == {2: -1, 3: 1}


def test_t_two_seven_theta_one_matches_odd_and_even_summands(
    theta_one_q_seven_data,
):
    r"""Check all four generic jumps for ``T(2,7)``, ``theta=1``.

    The odd part of the explicit theorem contributes the pair indexed by
    ``e=2``.  Its even part contributes the pair indexed by ``e=3``.  After
    converting from powers of ``xi=zeta^-1`` to ``zeta^a``, their signs are

    ``a=2 -> -1, a=3 -> +1, a=4 -> -1, a=5 -> +1``.
    """
    results = yanagida_generic_signature_jumps(theta_one_q_seven_data)

    assert {result.a: result.jump for result in results} == {
        2: -1,
        3: 1,
        4: -1,
        5: 1,
    }


def test_t_two_seven_theta_two_matches_the_explicit_decomposition(
    theta_two_q_seven_data,
):
    r"""A second character detects mistakes hidden by conjugate symmetry.

    For ``theta=2`` the generic roots are ``a=1,3,4,6``.  The explicit odd
    summands give negative signs on the first half-circle and positive signs
    at the conjugate roots.
    """
    results = yanagida_generic_signature_jumps(theta_two_q_seven_data)

    assert {result.a: result.jump for result in results} == {
        1: -1,
        3: -1,
        4: 1,
        6: 1,
    }


def test_local_result_exposes_the_exact_residue_data(theta_one_q_five_data):
    r"""The result explains the jump instead of returning an unexplained integer."""
    result = yanagida_local_signature_jump(theta_one_q_five_data, 2)

    assert isinstance(result, YanagidaSignatureJump)
    assert result.data is theta_one_q_five_data
    assert result.a == 2
    assert result.root == theta_one_q_five_data.zeta ** 2
    assert result.argument == QQ(2) / 5
    assert result.module_dimension == 1
    assert result.inertia == HermitianInertia(0, 1, 0)
    assert result.hodge_signs == (-1,)
    assert result.jump == -1
    assert result.residue_form == matrix(QQbar, [[QQbar(-4) / 5]])


def test_quotient_basis_really_annihilates_the_specialized_image(
    theta_one_q_five_data,
):
    r"""Columns representing the cokernel are chosen from ``ker(Theta^*)``."""
    data = theta_one_q_five_data
    result = yanagida_local_signature_jump(data, 2)
    local = data.local_model(2)
    theta_at_root = matrix(
        data.coefficient_field,
        data.m,
        data.m,
        lambda row, column: local.theta[row, column](local.root),
    )
    annihilator = (
        result.quotient_basis.conjugate_transpose() * theta_at_root
    )

    assert annihilator == matrix.zero(
        data.coefficient_field,
        data.m - 1,
        data.m,
    )


def test_minus_one_root_is_handled_without_floating_point_extension():
    r"""The ``xi=-1`` normalizer may require ``i`` outside the base field.

    For ``n=2`` the cyclotomic coefficient field is rational.  The residue
    nevertheless becomes a Hermitian form over ``QQbar`` after division by
    ``r_xi(xi)=-2i``.  In this example it is hyperbolic and has jump zero.
    """
    data = YanagidaTorusData(3, 2, (0, 0, 0))

    result = yanagida_local_signature_jump(data, 1)

    assert data.coefficient_field.degree() == 1
    assert result.root == -1
    assert result.inertia == HermitianInertia(1, 1, 0)
    assert result.jump == 0
    assert result.residue_form == result.residue_form.conjugate_transpose()


def test_odd_dimensional_yanagida_matrix_can_have_cancelling_hodge_signs():
    r"""A nontrivial local module need not produce a nonzero jump.

    For ``m=3,n=4,b=(0,1,3),a=2`` the local module has dimension two.  Its
    residue form has one positive and one negative direction, so the Witt
    class and signature jump vanish even though ``Theta`` has a two-dimensional
    cokernel at the root.
    """
    data = YanagidaTorusData(3, 4, (0, 1, 3))

    result = yanagida_local_signature_jump(data, 2)

    assert result.module_dimension == 2
    assert result.hodge_signs == (1, -1)
    assert result.jump == 0


@pytest.mark.parametrize(
    "m, n, b, a",
    [
        (2, 5, (1, 4), 2),
        (2, 7, (1, 6), 3),
        (3, 4, (0, 1, 3), 2),
        (3, 2, (0, 0, 0), 1),
    ],
)
def test_jump_parity_and_bound_follow_from_local_module_dimension(m, n, b, a):
    r"""A sum of ``m-1`` signs has fixed parity and absolute-value bound."""
    result = yanagida_local_signature_jump(YanagidaTorusData(m, n, b), a)

    assert abs(result.jump) <= m - 1
    assert result.jump % 2 == (m - 1) % 2
    assert result.inertia.nullity == 0


@pytest.mark.parametrize("a", [1, 4, 6, -1])
def test_exceptional_roots_are_not_filled_in_by_guessing(
    theta_one_q_five_data,
    a,
):
    r"""Equation (19) supplies no pairing at ``a=-b_i``.

    The values ``1`` and ``4`` modulo five are exceptional for ``b=(1,4)``.
    Integer lifts such as ``6`` and ``-1`` must be rejected in exactly the
    same way.
    """
    with pytest.raises(ValueError, match="not congruent to any -b_i"):
        yanagida_local_signature_jump(theta_one_q_five_data, a)


def test_trivial_root_is_rejected(theta_one_q_five_data):
    r"""Yanagida's local theorems assume ``a`` is nonzero modulo ``n``."""
    with pytest.raises(ValueError, match="nonzero modulo n"):
        yanagida_local_signature_jump(theta_one_q_five_data, 5)


@pytest.mark.parametrize("bad_data", [None, "T(2,5)", (2, 5, (1, 4))])
def test_signature_jump_requires_validated_yanagida_data(bad_data):
    """Raw tuples cannot carry the exact matrices and normalization metadata."""
    with pytest.raises(TypeError, match="YanagidaTorusData"):
        yanagida_local_signature_jump(bad_data, 2)


def test_generic_collection_is_sorted_and_omits_exceptional_roots(
    theta_one_q_seven_data,
):
    r"""The batch helper returns exactly the roots covered by Theorem 1.3."""
    results = yanagida_generic_signature_jumps(theta_one_q_seven_data)

    assert tuple(result.a for result in results) == (2, 3, 4, 5)
    assert tuple(result.argument for result in results) == (
        QQ(2) / 7,
        QQ(3) / 7,
        QQ(4) / 7,
        QQ(5) / 7,
    )


def test_generic_jumps_use_signature_function_half_jump_convention(
    theta_one_q_seven_data,
):
    r"""Crossing a computed weight changes ``SignatureFunction`` by twice it.

    This integration test guards against returning the full one-sided
    difference from the residue layer and then doubling it a second time in
    ``SignatureFunction``.
    """
    results = yanagida_generic_signature_jumps(theta_one_q_seven_data)
    signature = SignatureFunction(
        values=[(result.argument, result.jump) for result in results]
    )

    first = results[0]
    left_value = signature(first.argument - QQ(1) / 100)
    right_value = signature(first.argument + QQ(1) / 100)

    assert right_value - left_value == 2 * first.jump
    assert signature.total_sign_jump() == 0


def test_results_and_explanatory_matrices_are_immutable(theta_one_q_five_data):
    r"""The reported jump must remain synchronized with its residue form."""
    result = yanagida_local_signature_jump(theta_one_q_five_data, 2)

    with pytest.raises(FrozenInstanceError):
        result.a = 3
    with pytest.raises(ValueError, match="immutable"):
        result.residue_form[0, 0] = 1
    with pytest.raises(ValueError, match="immutable"):
        result.quotient_basis[0, 0] = 0


def test_signature_profile_certifies_vacuous_exceptional_roots(
    theta_one_q_five_data,
):
    r"""An excluded root need not contain any local module to pair.

    For ``T(2,5)`` and ``b=(1,4)``, the generic pairing theorem computes the
    jumps at ``a=2,3``.  At each excluded value ``a=1,4`` exactly one character
    coordinate equals ``-a``.  Here ``m=2``, so the dimension formula

    ``m - 1 - character_multiplicity = 2 - 1 - 1 = 0``

    proves that both exceptional primary modules vanish.  Their jumps are
    therefore genuinely zero; they are not zeros inserted as placeholders.
    """
    profile = yanagida_signature_profile(theta_one_q_five_data)

    assert isinstance(profile, YanagidaSignatureProfile)
    assert profile.data is theta_one_q_five_data
    assert tuple(result.a for result in profile.generic_jumps) == (2, 3)
    assert tuple(result.a for result in profile.exceptional_roots) == (1, 4)
    assert all(
        isinstance(result, YanagidaExceptionalRoot)
        for result in profile.exceptional_roots
    )
    assert all(
        result.character_multiplicity == 1
        and result.module_dimension == 0
        and result.has_zero_module
        and result.jump_is_resolved
        for result in profile.exceptional_roots
    )

    assert profile.unresolved_exceptional_roots == ()
    assert profile.zero_module_exceptional_roots == profile.exceptional_roots
    assert profile.is_complete_at_nontrivial_roots
    assert profile.require_complete_at_nontrivial_roots() is profile
    assert profile.known_jump_values == (
        (QQ(1) / 5, 0),
        (QQ(2) / 5, -1),
        (QQ(3) / 5, 1),
        (QQ(4) / 5, 0),
    )


def test_signature_profile_jump_lookup_reduces_integer_lifts(
    theta_one_q_five_data,
):
    r"""Lookup follows the same residue-class convention as local models.

    The assertions cover both kinds of resolved roots.  The lift ``7`` is the
    generic class ``a=2`` and must return the computed negative Hodge sign;
    the lift ``-1`` is the zero-module exceptional class ``a=4`` and must
    return its proved zero contribution.
    """
    profile = yanagida_signature_profile(theta_one_q_five_data)

    assert profile.jump_at(7) == -1
    assert profile.jump_at(-1) == 0


def test_signature_profile_reports_surviving_exceptional_modules():
    r"""Positive-dimensional exceptional modules remain visibly unresolved.

    For ``m=3,n=4,b=(0,1,3)``, the roots ``a=1`` and ``a=3`` each match one
    character coordinate.  Specializing equation (14) gives rank two and a
    one-dimensional cokernel at both roots.  Equation (19) is unavailable
    there, so the profile records the dimensions without assigning signs.

    The only generic root is ``a=2``.  Its two Hodge signs cancel, which is a
    computed zero and therefore appears in ``known_jump_values``.  This makes
    the distinction between "computed zero" and "unknown" testable.
    """
    data = YanagidaTorusData(3, 4, (0, 1, 3))

    profile = yanagida_signature_profile(data)

    assert tuple(result.a for result in profile.generic_jumps) == (2,)
    assert profile.generic_jumps[0].jump == 0
    assert profile.zero_module_exceptional_roots == ()
    assert tuple(
        (result.a, result.character_multiplicity, result.module_dimension)
        for result in profile.unresolved_exceptional_roots
    ) == ((1, 1, 1), (3, 1, 1))
    assert not profile.is_complete_at_nontrivial_roots
    assert profile.known_jump_values == ((QQ(1) / 2, 0),)

    with pytest.raises(NotImplementedError, match=r"a=1.*dimension 1"):
        profile.jump_at(1)
    with pytest.raises(NotImplementedError, match=r"a=\(1, 3\)"):
        profile.require_complete_at_nontrivial_roots()


def test_signature_profile_uses_character_multiplicity_not_just_exceptionality():
    r"""Repeated exceptional coordinates can make the local module vanish.

    In ``b=(1,1,2)`` modulo four, ``a=2`` matches the single coordinate ``2``
    and leaves a one-dimensional unresolved module.  In contrast, ``a=3``
    matches both coordinates equal to ``1`` and leaves dimension zero.  A
    boolean exceptional/nonexceptional flag would lose precisely this
    distinction, so the public records expose the full multiplicity.
    """
    data = YanagidaTorusData(3, 4, (1, 1, 2))

    profile = yanagida_signature_profile(data)
    exceptional_by_a = {
        result.a: result for result in profile.exceptional_roots
    }

    assert tuple(result.a for result in profile.generic_jumps) == (1,)
    assert exceptional_by_a[2].character_multiplicity == 1
    assert exceptional_by_a[2].module_dimension == 1
    assert not exceptional_by_a[2].jump_is_resolved
    assert exceptional_by_a[3].character_multiplicity == 2
    assert exceptional_by_a[3].module_dimension == 0
    assert exceptional_by_a[3].jump_is_resolved
    assert profile.jump_at(3) == 0


def test_signature_profile_does_not_claim_coverage_at_t_one(
    theta_one_q_five_data,
):
    r"""Completeness at nontrivial roots must not be confused with global data.

    Yanagida's Theorems 1.2 and 1.3 both impose ``a != 0``.  Even though the
    ``T(2,5)`` profile resolves all four nonzero fifth roots, asking for the
    class ``a=0`` must retain that mathematical boundary rather than using
    periodic reduction to fabricate a value at ``t=1``.
    """
    profile = yanagida_signature_profile(theta_one_q_five_data)
    assert profile.is_complete_at_nontrivial_roots

    with pytest.raises(ValueError, match="nonzero modulo n"):
        profile.jump_at(0)
    with pytest.raises(ValueError, match="nonzero modulo n"):
        profile.jump_at(5)


@pytest.mark.parametrize("bad_data", [None, "T(2,5)", (2, 5, (1, 4))])
def test_signature_profile_requires_validated_matrix_data(bad_data):
    r"""The profile needs exact ``Theta`` and ``Psi`` matrices, not raw labels."""
    with pytest.raises(TypeError, match="YanagidaTorusData"):
        yanagida_signature_profile(bad_data)


def test_signature_profile_and_exceptional_records_are_immutable():
    r"""Coverage cannot be mutated after downstream code has checked it."""
    profile = yanagida_signature_profile(
        YanagidaTorusData(3, 4, (0, 1, 3))
    )

    with pytest.raises(FrozenInstanceError):
        profile.exceptional_roots = ()
    with pytest.raises(FrozenInstanceError):
        profile.exceptional_roots[0].module_dimension = 0
