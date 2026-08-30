#!/usr/bin/env sage -python

r"""Local twisted-signature jumps from Yanagida's torus-knot matrices.

The module :mod:`gaknot.invariants.torus_twisted_blanchfield` implements the
local module presentation ``Theta_b(a)`` and the local Blanchfield numerator
``t^n Psi_b(a)``.  This module performs the next step: it extracts the
signature jump at the generic root

.. math::

    \xi = \exp(2\pi i a/n).

The construction follows Algorithm 2.6 of Borodzik--Conway--Politarczyk,
*Twisted Blanchfield pairings and twisted signatures III: Applications*.
That paper defines the complex signature jump as the sum of the signs
``epsilon`` of the odd elementary forms.  This is also the convention used by
``SignatureFunction``: a stored jump ``j`` changes the signature by ``2*j``.

Why the generic Yanagida case is simpler
-----------------------------------------

At a generic root, i.e. when ``a`` is nonzero and is not congruent to any
``-b_i``, Yanagida's presentation has

``rank(Theta_b(a)(xi)) = 1``

and

``ord_xi(det(Theta_b(a))) = m - 1``.

It follows that the local module is a direct sum of ``m-1`` copies of
``O_xi/(t-xi)``.  Every elementary divisor therefore has exponent one.  No
higher-order cyclic orthogonalisation is necessary: all Hodge signs are the
inertia signs of one residue Hermitian form on

``coker(Theta_b(a)(xi))``.

Yanagida presents the pairing with values in ``O_xi/(Delta)``.  The standard
map to the linking-form target ``Omega_xi/O_xi`` divides the numerator matrix
by ``Delta``.  Since ``Delta`` vanishes to order ``m-1`` and the numerator
vanishes to order ``m-2`` on the local module, this quotient has a simple
pole.  Multiplying by ``t-xi`` and evaluating at ``xi`` gives its principal
residue form.

Finally, a residue by itself has a root-dependent complex phase.  Dividing by
the value at ``xi`` of the explicit ``xi``-positive polynomial from Example
2.3 of the Applications paper turns it into an ordinary Hermitian matrix.
Its inertia is calculated over ``QQbar`` using exact algebraic arithmetic;
no floating-point eigenvalue threshold enters the result.

Current scope
-------------

Yanagida's pairing formula assumes
``a not in {0, -b_1, ..., -b_m}``.  This module enforces that hypothesis.
Equation (14) still presents the local module at exceptional nonzero roots,
but equation (19) does not provide the pairing there, so this module does not
guess the missing jumps.  Consequently :func:`yanagida_generic_signature_jumps`
returns the complete collection of jumps supplied by Yanagida's theorem, not
necessarily the complete twisted signature function.
"""

from dataclasses import dataclass, field

from sage.all import AA, I, Integer, QQ, QQbar, identity_matrix, matrix

from gaknot.invariants.torus_twisted_blanchfield import (
    YanagidaTorusData,
    is_regular_at,
)


def _immutable(matrix_value):
    """Mark a Sage matrix immutable and return it."""
    matrix_value.set_immutable()
    return matrix_value


def _adjoint(matrix_value):
    """Return the conjugate transpose of a matrix over an algebraic field."""
    return matrix_value.conjugate_transpose()


@dataclass(frozen=True)
class HermitianInertia:
    r"""Exact positive, negative and zero indices of a Hermitian matrix."""

    positive_index: object
    negative_index: object
    nullity: object

    def __post_init__(self):
        for name in ("positive_index", "negative_index", "nullity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, Integer)):
                raise TypeError(f"{name} must be an integer.")
            value = Integer(value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)

    @property
    def signature(self):
        """Return ``positive_index - negative_index``."""
        return self.positive_index - self.negative_index

    @property
    def dimension(self):
        """Return the total dimension recorded by the inertia."""
        return self.positive_index + self.negative_index + self.nullity


def exact_hermitian_inertia(matrix_value):
    r"""Return the exact inertia of a Hermitian matrix over ``QQbar``.

    The calculation uses Hermitian congruence and Schur complements.  A
    nonzero diagonal entry supplies a one-dimensional pivot.  If all diagonal
    entries vanish but an off-diagonal entry ``z`` is nonzero, replacing
    ``e_i`` by ``e_i + conjugate(z)e_j`` creates the positive diagonal entry
    ``2*|z|^2``.  Thus the algorithm also handles hyperbolic blocks without
    numerical eigenvalues.

    The sign of each pivot is decided in Sage's exact real algebraic field
    ``AA``.  Therefore a very small but nonzero algebraic eigenvalue is never
    mistaken for zero because of a tolerance.
    """
    if matrix_value.nrows() != matrix_value.ncols():
        raise ValueError("The matrix must be square.")

    size = matrix_value.nrows()
    working = matrix(
        QQbar,
        size,
        size,
        lambda row, column: QQbar(matrix_value[row, column]),
    )
    if working != _adjoint(working):
        raise ValueError("The matrix must be Hermitian.")

    positive = Integer(0)
    negative = Integer(0)
    zero = Integer(0)

    while working.nrows() > 0:
        size = working.nrows()
        pivot_index = next(
            (index for index in range(size) if working[index, index] != 0),
            None,
        )

        if pivot_index is None:
            off_diagonal = next(
                (
                    (row, column)
                    for row in range(size)
                    for column in range(row + 1, size)
                    if working[row, column] != 0
                ),
                None,
            )
            if off_diagonal is None:
                # A Hermitian matrix with no diagonal or off-diagonal entries
                # is the zero matrix.  Its entire remaining space is radical.
                zero += size
                break

            row, column = off_diagonal
            z = working[row, column]
            change_of_basis = identity_matrix(QQbar, size)
            # The replacement vector is e_row + conjugate(z)e_column.
            # Its self-pairing is 2*z*conjugate(z), which is strictly positive.
            change_of_basis[column, row] = z.conjugate()
            working = (
                _adjoint(change_of_basis)
                * working
                * change_of_basis
            )
            pivot_index = row

        pivot = working[pivot_index, pivot_index]
        if pivot != pivot.conjugate():
            raise ArithmeticError("A Hermitian pivot was not exactly real.")

        # ``real()`` lands in AA and its sign comparison is exact.
        pivot_sign = AA(pivot.real()).sign()
        if pivot_sign > 0:
            positive += 1
        elif pivot_sign < 0:
            negative += 1
        else:
            # This branch should be unreachable because pivot_index was chosen
            # from an exactly nonzero entry, but retain the guard explicitly.
            raise ArithmeticError("A nonzero algebraic pivot had zero sign.")

        remaining = [
            index for index in range(size) if index != pivot_index
        ]
        working = matrix(
            QQbar,
            len(remaining),
            len(remaining),
            lambda row, column: (
                working[remaining[row], remaining[column]]
                - working[remaining[row], pivot_index]
                * working[pivot_index, remaining[column]]
                / pivot
            ),
        )

        if working != _adjoint(working):
            raise ArithmeticError("The Hermitian Schur complement lost symmetry.")

    return HermitianInertia(positive, negative, zero)


def _xi_positive_residue_value(root, a, n):
    r"""Return ``r_xi(xi)`` for the standard ``xi``-positive polynomial.

    For a nonreal root, Example 2.3 permits ``1-xi*t`` in the upper half-plane
    and its negative in the lower half-plane.  At ``xi=-1`` it permits
    ``i(t-1)``.  The root ``xi=1`` corresponds to ``a=0`` and is outside
    Yanagida's hypotheses.
    """
    xi = QQbar(root)
    if 2 * a < n:
        return 1 - xi ** 2
    if 2 * a > n:
        return xi ** 2 - 1
    return QQbar(-2 * I)


def _specialize_regular_matrix(matrix_value, root, coefficient_field):
    """Evaluate a rational matrix whose entries are regular at ``root``."""
    for entry in matrix_value.list():
        if not is_regular_at(entry, root):
            raise ArithmeticError(
                "A matrix entry has a pole at the localization root."
            )
    return matrix(
        coefficient_field,
        matrix_value.nrows(),
        matrix_value.ncols(),
        lambda row, column: matrix_value[row, column](root),
    )


@dataclass(frozen=True)
class YanagidaSignatureJump:
    r"""Exact local signature data at one generic ``n``-th root.

    ``quotient_basis`` contains columns representing a basis of
    ``coker(Theta_b(a)(root))``.  ``residue_form`` is the corresponding
    normalized Hermitian form over ``QQbar``.  Its positive and negative
    inertia signs are precisely the Hodge signs of the exponent-one elementary
    forms; their sum with signs is exposed as :attr:`jump`.
    """

    data: YanagidaTorusData = field(repr=False)
    a: object
    root: object = field(repr=False)
    argument: object
    quotient_basis: object = field(repr=False)
    residue_form: object = field(repr=False)
    inertia: HermitianInertia

    @property
    def jump(self):
        """Return the half-jump stored by ``SignatureFunction``."""
        return self.inertia.signature

    @property
    def module_dimension(self):
        """Return the dimension of the first-order local module."""
        return self.inertia.dimension

    @property
    def hodge_signs(self):
        """Return one ``+1`` or ``-1`` for each nondegenerate summand."""
        return (
            (Integer(1),) * self.inertia.positive_index
            + (Integer(-1),) * self.inertia.negative_index
        )


def yanagida_local_signature_jump(data, a):
    r"""Compute the twisted-signature jump at ``exp(2*pi*i*a/n)``.

    Args:
        data: A :class:`YanagidaTorusData` object containing ``m``, ``n`` and
              the character orbit ``b``.
        a: A nonzero integer residue class modulo ``n`` satisfying
           ``a != -b_i`` for every orbit coordinate.

    Returns:
        An immutable :class:`YanagidaSignatureJump` with the normalized residue
        form, exact inertia and half-jump.

    Raises:
        TypeError: If ``data`` is not a ``YanagidaTorusData`` object or ``a``
                   is not an exact integer (delegated to the local model).
        ValueError: If the root is trivial or exceptional for Yanagida's
                    pairing formula.
        ArithmeticError: If a claimed regularity, rank or nondegeneracy
                         identity fails.
    """
    if not isinstance(data, YanagidaTorusData):
        raise TypeError("data must be a YanagidaTorusData object.")

    local = data.local_model(a)
    pairing = local.pairing

    # Specialize the presentation.  At a generic root its image has dimension
    # one, so the cokernel has the expected dimension m-1.
    theta_at_root = _specialize_regular_matrix(
        local.theta,
        local.root,
        data.coefficient_field,
    )
    if theta_at_root.rank() != 1:
        raise ArithmeticError(
            "Generic Theta specialization should have rank one."
        )

    # Rows of the left kernel annihilate im(Theta).  Their conjugate
    # transposes therefore give column representatives for a complementary
    # basis of the cokernel.
    left_kernel_basis = theta_at_root.left_kernel().basis_matrix()
    quotient_basis = left_kernel_basis.conjugate_transpose()
    if quotient_basis.ncols() != data.m - 1:
        raise ArithmeticError(
            "The specialized cokernel should have dimension m-1."
        )

    # Embed the O/(Delta)-valued pairing in Omega/O by division by Delta, then
    # remove its expected simple pole.  The symmetric order is convenient
    # because it is compatible with the involution.  It differs from the
    # theorem's displayed order by t^(n(m-1)), whose value at every n-th root
    # is one, so both representatives produce the same residue.
    linking_matrix = pairing.matrix / pairing.symmetric_order
    principal_matrix = (data.t - local.root) * linking_matrix
    principal_residue = _specialize_regular_matrix(
        principal_matrix,
        local.root,
        data.coefficient_field,
    )
    restricted_residue = (
        quotient_basis.conjugate_transpose()
        * principal_residue
        * quotient_basis
    )

    # Divide out the phase of the standard xi-positive polynomial.  Coercing
    # to QQbar also handles xi=-1, where the normalizer uses i even if i is not
    # contained in the original cyclotomic coefficient field.
    normalizer = _xi_positive_residue_value(
        local.root,
        local.a,
        data.n,
    )
    residue_form = matrix(
        QQbar,
        restricted_residue.nrows(),
        restricted_residue.ncols(),
        lambda row, column: (
            QQbar(restricted_residue[row, column]) / normalizer
        ),
    )
    if residue_form != residue_form.conjugate_transpose():
        raise ArithmeticError(
            "The normalized principal residue is not Hermitian."
        )

    inertia = exact_hermitian_inertia(residue_form)
    if inertia.nullity != 0:
        raise ArithmeticError(
            "The normalized local residue form should be nondegenerate."
        )

    return YanagidaSignatureJump(
        data=data,
        a=local.a,
        root=local.root,
        argument=QQ(local.a) / data.n,
        quotient_basis=_immutable(quotient_basis),
        residue_form=_immutable(residue_form),
        inertia=inertia,
    )


def yanagida_generic_signature_jumps(data):
    r"""Return all signature jumps covered by Yanagida's generic formula.

    Results are ordered by ``a=1,...,n-1`` and exceptional roots are omitted.
    The omission is mathematical rather than computational: equation (19) of
    Yanagida's paper assumes that no orbit coordinate equals ``-a``.
    """
    if not isinstance(data, YanagidaTorusData):
        raise TypeError("data must be a YanagidaTorusData object.")

    results = []
    for a in range(1, data.n):
        local = data.local_model(a)
        if local.is_generic:
            results.append(yanagida_local_signature_jump(data, a))
    return tuple(results)
