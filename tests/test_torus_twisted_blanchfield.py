#!/usr/bin/env sage -python

r"""Tests for Yanagida's explicit torus-knot Blanchfield matrices.

These tests deliberately separate four layers of the construction:

* arithmetic input and the canonical Bezout convention;
* the global representation matrices ``C``, ``X``, ``Y`` and ``M``;
* the root-dependent module presentation ``Theta_b(a)``;
* the generic local pairing matrix ``t^n Psi_b(a)``.

That separation matters because equation (14) of Yanagida's paper applies at
every nonzero ``n``-th root, including roots for which some ``b_i = -a``,
whereas equation (19) presents the pairing only after those exceptional roots
have been removed.  A passing test should therefore tell us not only that a
matrix has the expected entries, but also that the public API enforces the
correct theorem at the correct root.

All comparisons take place in exact cyclotomic fields and rational-function
fields.  No numerical approximation is used to decide whether a denominator
vanishes or whether two matrix formulas agree.
"""

from dataclasses import FrozenInstanceError

import pytest
from sage.all import Infinity, Integer, identity_matrix, matrix

from gaknot import YanagidaTorusData
from gaknot.invariants.torus_twisted_blanchfield import (
    YanagidaLocalModel,
    YanagidaLocalPairing,
    canonical_bezout_coefficients,
    is_local_unit,
    is_regular_at,
    local_valuation,
    yanagida_adjoint,
    yanagida_involution,
)


@pytest.fixture(scope="module")
def two_by_five_data():
    r"""Return a small orbit having both generic and exceptional roots.

    The orbit ``(1,4)`` sums to zero in ``Z/5Z``.  Its exceptional nonzero
    localization parameters are ``-1 = 4`` and ``-4 = 1``.  Thus ``a=2`` is
    generic, while ``a=4`` lets us test the projection in ``Theta`` without
    constructing a different global representation.
    """
    return YanagidaTorusData(2, 5, (1, 4))


@pytest.fixture(scope="module")
def three_by_four_data():
    r"""Return an odd-dimensional example for the parity-sensitive factor.

    Here ``delta_3(t)=1`` and ``a=2`` avoids the exceptional values ``1`` and
    ``3``.  Together with the even-dimensional fixture above, this exercises
    both exact encodings of ``t^(n(1-m)/2) delta_m(t)``.
    """
    return YanagidaTorusData(3, 4, (0, 1, 3))


@pytest.mark.parametrize(
    "m, n, expected",
    [
        (2, 3, (-1, 1)),
        (2, 5, (-2, 1)),
        (3, 4, (-1, 1)),
        (4, 5, (-1, 1)),
        (5, 7, (-4, 3)),
    ],
)
def test_canonical_bezout_coefficients_use_yanagida_ranges(m, n, expected):
    r"""The helper must choose Yanagida's pair, not any Bezout solution.

    Adding ``n`` to ``r`` and subtracting ``m`` from ``s`` preserves the
    identity ``m*r+n*s=1`` but changes the powers in ``M=X^sY^r``.  The range
    conditions are consequently part of the mathematical input, not merely a
    cosmetic normalization.
    """
    r, s = canonical_bezout_coefficients(m, n)

    assert (r, s) == expected
    assert m * r + n * s == 1
    assert -n < r < 0 < s < m


@pytest.mark.parametrize("m, n", [(1, 3), (3, 1), (0, 5), (-2, 3)])
def test_canonical_bezout_coefficients_reject_small_parameters(m, n):
    """A torus *knot* input in the paper requires both parameters above one."""
    with pytest.raises(ValueError, match="greater than one"):
        canonical_bezout_coefficients(m, n)


@pytest.mark.parametrize("m, n", [(2, 4), (3, 6), (6, 9)])
def test_canonical_bezout_coefficients_reject_noncoprime_parameters(m, n):
    """Without coprimality there is no Bezout identity equal to one."""
    with pytest.raises(ValueError, match="coprime"):
        canonical_bezout_coefficients(m, n)


@pytest.mark.parametrize(
    "m, n",
    [(True, 5), (2.0, 5), (2, False), (2, "5")],
)
def test_canonical_bezout_coefficients_reject_inexact_integer_spellings(m, n):
    """Booleans, floating-point values and strings must not become exponents."""
    with pytest.raises(TypeError, match="must be an integer"):
        canonical_bezout_coefficients(m, n)


def test_orbit_coordinates_are_normalized_modulo_n():
    r"""Equivalent integer lifts should produce one canonical orbit tuple.

    ``6`` and ``-1`` represent ``1`` and ``4`` in ``Z/5Z``.  Normalizing at
    the boundary makes equality, exceptional-root detection and error
    messages independent of which lifts a caller happened to choose.
    """
    data = YanagidaTorusData(2, 5, [6, -1])

    assert data.b == (1, 4)
    assert all(isinstance(coordinate, Integer) for coordinate in data.b)


@pytest.mark.parametrize(
    "b, error_type, message",
    [
        ("1,4", TypeError, "list or tuple"),
        ((1,), ValueError, "exactly m=2"),
        ((1, 4, 0), ValueError, "exactly m=2"),
        ((1, 3), ValueError, r"sum\(b_i\) = 0"),
        ((True, 4), TypeError, r"b\[0\] must be an integer"),
        ((1.0, 4), TypeError, r"b\[0\] must be an integer"),
    ],
)
def test_character_orbit_validation_rejects_malformed_data(b, error_type, message):
    r"""The representation needs exactly ``m`` integral coordinates of sum zero."""
    with pytest.raises(error_type, match=message):
        YanagidaTorusData(2, 5, b)


@pytest.mark.parametrize("fixture_name", ["two_by_five_data", "three_by_four_data"])
def test_global_matrices_satisfy_their_defining_relations(request, fixture_name):
    r"""Check the algebra from which all later matrices are assembled.

    The cyclic shift has ``C^m=tI``.  Therefore ``X=C^n`` satisfies
    ``X^m=t^nI``.  The diagonal entries of ``Y`` are ``t*zeta_n^b_i``, so
    ``Y^n=t^nI`` as well.  Finally ``M`` must use the canonical powers in the
    order ``X^sY^r``.  A transpose, reversed product or off-by-one shift would
    be detected here before producing a much less readable failure in
    ``Psi``.
    """
    data = request.getfixturevalue(fixture_name)
    identity = identity_matrix(data.function_field, data.m)

    assert data.C ** data.m == data.t * identity
    assert data.X == data.C ** data.n
    assert data.X ** data.m == data.t ** data.n * identity
    assert data.Y ** data.n == data.t ** data.n * identity
    assert data.M == data.X ** data.s * data.Y ** data.r


def test_even_m_prefactor_eliminates_half_powers(two_by_five_data):
    r"""For ``m=2,n=5`` the combined delta factor is ``t^-2/(t-1)``.

    Indeed, ``t^(5(1-2)/2)/(t^(1/2)-t^(-1/2))`` simplifies to that rational
    function.  This test ensures the implementation stays inside ``K(t)``
    rather than accidentally introducing a symbolic square root of ``t``.
    """
    data = two_by_five_data

    assert data.symmetric_prefactor == data.t ** (-2) / (data.t - 1)
    assert data.symmetric_prefactor.parent() is data.function_field


def test_odd_m_prefactor_is_an_integral_power(three_by_four_data):
    r"""For odd ``m``, ``delta_m=1`` and no denominator is introduced."""
    data = three_by_four_data

    assert data.symmetric_prefactor == data.t ** (-4)


def test_zero_surgery_order_uses_equation_18(two_by_five_data):
    r"""Record the exact global normalization used by Yanagida's Remark 5.12.

    The factor ``1-t`` belongs in the denominator for the zero-surgery order.
    This is intentionally distinct from the existing knot-exterior twisted
    Alexander function in the package.
    """
    data = two_by_five_data
    expected_denominator = (1 - data.t)
    for coordinate in data.b:
        expected_denominator *= 1 - data.zeta ** coordinate * data.t
    expected = (1 - data.t ** data.n) ** (data.m - 1) / expected_denominator

    assert data.zero_surgery_twisted_alexander_order == expected


def test_generic_projection_is_identity_and_theta_is_equation_14(two_by_five_data):
    r"""At a generic root, equation (14) reduces to its geometric-sum block.

    For ``a=2`` neither coordinate of ``b=(1,4)`` equals ``-2=3``.  Hence
    ``P=I`` and the two outer projections in ``Theta`` leave the matrix
    polynomial untouched.
    """
    data = two_by_five_data
    local = data.local_model(2)
    identity = identity_matrix(data.function_field, data.m)
    geometric_sum = sum(
        (data.X ** exponent for exponent in range(data.m)),
        matrix.zero(data.function_field, data.m, data.m),
    )
    expected = data.t ** (-data.n) * geometric_sum * data.X

    assert local.is_generic
    assert local.projection == identity
    assert local.theta == expected


def test_exceptional_projection_and_theta_keep_equation_14(two_by_five_data):
    r"""Equation (14) remains available when one character weight is exceptional.

    At ``a=4`` we have ``-a=1``, so the first orbit coordinate is deleted and
    ``P=diag(0,1)``.  The complementary ``I-P`` term is essential: it makes
    the first coordinate a unit-presented summand while the projected
    geometric block controls the remaining coordinate.
    """
    data = two_by_five_data
    local = data.local_model(4)
    identity = identity_matrix(data.function_field, data.m)
    expected_projection = matrix.diagonal(data.function_field, [0, 1])
    geometric_sum = sum(
        (data.X ** exponent for exponent in range(data.m)),
        matrix.zero(data.function_field, data.m, data.m),
    )
    expected_theta = (
        identity - expected_projection
        + expected_projection
        * (data.t ** (-data.n) * geometric_sum * data.X)
        * expected_projection
    )

    assert not local.is_generic
    assert local.projection == expected_projection
    assert local.theta == expected_theta


@pytest.mark.parametrize("a", [2, 7, -3, Integer(12)])
def test_local_parameter_is_normalized_modulo_n(two_by_five_data, a):
    r"""All integer lifts of ``a=2`` must select the same root and matrix."""
    local = two_by_five_data.local_model(a)
    reference = two_by_five_data.local_model(2)

    assert local.a == 2
    assert local.root == reference.root
    assert local.projection == reference.projection
    assert local.theta == reference.theta


@pytest.mark.parametrize("a", [0, 5, -5, Integer(10)])
def test_local_model_rejects_the_trivial_root(two_by_five_data, a):
    r"""Theorems 1.2 and 1.3 both assume ``a`` is nonzero in ``Z/nZ``."""
    with pytest.raises(ValueError, match="nonzero modulo n"):
        two_by_five_data.local_model(a)


@pytest.mark.parametrize("a", [True, 2.0, "2", None])
def test_local_model_rejects_nonintegral_parameters(two_by_five_data, a):
    """A localization root must be selected by an exact residue class."""
    with pytest.raises(TypeError, match="a must be an integer"):
        two_by_five_data.local_model(a)


def test_theta_has_the_expected_order_up_to_the_sign_unit(two_by_five_data):
    r"""The exact determinant agrees with the paper's order ideal.

    With the displayed cyclic-shift matrix, direct determinant expansion gives

    ``det(Theta)=(-1)^(m-1)t^(n(1-m))(1-t^n)^(m-1)``.

    Equation (17) suppresses ``(-1)^(m-1)``, as is legitimate for a module
    order because that sign is a unit.  Testing the exact sign here prevents a
    matrix-convention change from being mistaken for an algebra error.
    """
    data = two_by_five_data
    theta = data.local_model(2).theta
    normalized_order = (
        data.t ** (data.n * (1 - data.m))
        * (1 - data.t ** data.n) ** (data.m - 1)
    )

    assert theta.det() == (-1) ** (data.m - 1) * normalized_order
    assert theta.det() / normalized_order in (-1, 1)


@pytest.mark.parametrize(
    "fixture_name, a",
    [("two_by_five_data", 2), ("three_by_four_data", 2)],
)
def test_generic_theta_specialization_has_nullity_m_minus_one(
    request,
    fixture_name,
    a,
):
    r"""Specializing at the chosen root detects the local primary summand.

    At a generic ``n``-th root, the geometric sum
    ``I+X+...+X^(m-1)`` has rank one.  Thus ``Theta(zeta_n^a)`` has nullity
    ``m-1``, matching the exponent of ``1-t^n`` in its determinant/order.
    """
    data = request.getfixturevalue(fixture_name)
    local = data.local_model(a)
    specialized = matrix(
        data.coefficient_field,
        data.m,
        data.m,
        lambda row, column: local.theta[row, column](local.root),
    )

    assert specialized.rank() == 1
    assert specialized.right_nullity() == data.m - 1


def test_psi_is_exactly_equation_19(two_by_five_data):
    r"""Reassemble equation (19) from the stored representation matrices.

    This catches the three easiest transcription errors: using ``X^s`` in
    place of ``X^-s``, inverting ``I-M`` rather than ``I-M^-1``, or replacing
    the final ``Y^-r`` by ``Y^r``.
    """
    data = two_by_five_data
    identity = identity_matrix(data.function_field, data.m)
    expected = (
        data.symmetric_prefactor
        * (1 - data.t ** data.n) ** (data.m - 2)
        * (identity - data.X ** (-data.s))
        * (identity - data.M ** (-1)).inverse()
        * (identity - data.Y ** (-data.r))
    )

    assert data.psi == expected


def test_pairing_is_available_at_a_generic_root(two_by_five_data):
    r"""Theorem 1.3 packages ``Theta`` and ``t^n Psi`` over the same root."""
    data = two_by_five_data
    local = data.local_model(2)
    pairing = local.pairing

    assert isinstance(pairing, YanagidaLocalPairing)
    assert pairing.a == local.a
    assert pairing.root == local.root
    assert pairing.presentation is local.theta
    assert pairing.matrix == data.t ** data.n * data.psi


@pytest.mark.parametrize("a", [1, 4])
def test_pairing_is_rejected_at_exceptional_roots(two_by_five_data, a):
    r"""Do not extend Theorem 1.3 beyond its stated invertibility hypothesis.

    ``Theta`` still exists at these roots by Theorem 1.2.  Only asking for the
    pairing must fail, and the diagnostic identifies the relevant character
    weights rather than reporting a low-level singular-matrix error.
    """
    local = two_by_five_data.local_model(a)

    assert isinstance(local, YanagidaLocalModel)
    assert local.theta.nrows() == two_by_five_data.m
    with pytest.raises(ValueError, match="not congruent to any -b_i"):
        _ = local.pairing


def test_local_orders_differ_by_the_expected_unit(two_by_five_data):
    r"""The theorem and symmetric representatives generate one local ideal.

    Replacing the exponent ``n(1-m)/2`` by ``n(m-1)/2`` multiplies the order
    by ``t^(n(m-1))``.  Since the localization root is nonzero, this monomial
    is a unit and the quotient module is unchanged.
    """
    data = two_by_five_data
    pairing = data.local_model(2).pairing
    ratio = pairing.order / pairing.symmetric_order

    assert ratio == data.t ** (data.n * (data.m - 1))
    assert is_local_unit(ratio, pairing.root)


def test_global_and_symmetric_orders_are_locally_associated(two_by_five_data):
    r"""Verify the local-unit comparison made in Yanagida's Remark 5.12.

    At a generic root every denominator in equation (18) is a unit.  The
    global zero-surgery order and equation (17) should consequently differ by
    valuation zero even though their global rational functions look quite
    different.
    """
    data = two_by_five_data
    pairing = data.local_model(2).pairing
    quotient = (
        data.zero_surgery_twisted_alexander_order / pairing.symmetric_order
    )

    assert is_local_unit(quotient, pairing.root)
    assert local_valuation(pairing.symmetric_order, pairing.root) == data.m - 1


def test_psi_entries_are_regular_at_the_generic_root(two_by_five_data):
    r"""Equation (19) must define germs, not merely formal rational functions."""
    data = two_by_five_data
    local = data.local_model(2)

    assert all(is_regular_at(entry, local.root) for entry in data.psi.list())


def test_yanagida_involution_conjugates_coefficients_and_inverts_t(
    two_by_five_data,
):
    r"""Test both parts of ``f#=conjugate(f(t^-1))`` independently of matrices."""
    data = two_by_five_data
    value = data.zeta * data.t ** 2 / (1 - data.zeta ** 2 * data.t)
    expected = (
        data.zeta ** (-1)
        * data.t ** (-2)
        / (1 - data.zeta ** (-2) * data.t ** (-1))
    )

    assert yanagida_involution(value) == expected


def test_psi_satisfies_yanagidas_adjoint_identity(two_by_five_data):
    r"""Check the identity stated immediately after equation (19).

    Remark 5.14 states ``(t^n Psi)^(# transpose)=Psi``.  This is stronger and
    more informative than checking a few entries numerically: it verifies the
    coefficient conjugation, transposition, power of ``t`` and matrix order
    simultaneously.
    """
    data = two_by_five_data
    pairing_matrix = data.t ** data.n * data.psi

    assert yanagida_adjoint(pairing_matrix) == data.psi


def test_pairing_hermitian_defect_is_divisible_by_the_local_order(
    two_by_five_data,
):
    r"""Verify the exact identity that makes the quotient pairing Hermitian.

    The numerator matrix is not literally Hermitian over the rational-function
    field.  Remark 5.14 says its defect is the symmetric order multiplied by
    the final matrix factor in ``Psi``.  It therefore vanishes in the quotient
    by that order, which is precisely the Hermitian statement needed for the
    Blanchfield form.
    """
    data = two_by_five_data
    pairing = data.local_model(2).pairing
    identity = identity_matrix(data.function_field, data.m)
    final_factor = (
        (identity - data.X ** (-data.s))
        * (identity - data.M ** (-1)).inverse()
        * (identity - data.Y ** (-data.r))
    )
    expected_defect = pairing.symmetric_order * final_factor

    assert yanagida_adjoint(pairing.matrix) - pairing.matrix == expected_defect


def test_local_valuation_distinguishes_regular_elements_units_and_poles(
    two_by_five_data,
):
    r"""Exercise the exact algebra used as a stand-in for the analytic local ring."""
    data = two_by_five_data
    root = data.zeta ** 2
    local_parameter = data.t - root
    unit = 1 + local_parameter

    assert local_valuation(local_parameter ** 3 * unit, root) == 3
    assert local_valuation(unit / local_parameter, root) == -1
    assert local_valuation(data.function_field.zero(), root) == Infinity
    assert is_regular_at(local_parameter, root)
    assert not is_local_unit(local_parameter, root)
    assert is_local_unit(unit, root)
    assert not is_regular_at(1 / local_parameter, root)


def test_public_models_and_their_matrices_are_immutable(two_by_five_data):
    r"""Prevent callers from invalidating identities cached inside the models.

    Frozen dataclasses protect attributes, while Sage's immutable flag protects
    matrix entries.  Both layers are tested because either one alone would
    still permit a partially mutated object.
    """
    data = two_by_five_data
    local = data.local_model(2)

    with pytest.raises(FrozenInstanceError):
        data.m = 7
    with pytest.raises(FrozenInstanceError):
        local.a = 3
    with pytest.raises(ValueError, match="immutable"):
        data.X[0, 0] = 42
    with pytest.raises(ValueError, match="immutable"):
        local.theta[0, 0] = 42
