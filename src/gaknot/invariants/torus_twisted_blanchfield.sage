#!/usr/bin/env sage -python

r"""Yanagida's local twisted Blanchfield matrices for torus knots.

This module implements the explicit matrices in Koki Yanagida,
*Blanchfield pairings and twisted Blanchfield pairings of torus knots*,
arXiv:2602.07575v2.  The notation follows Sections 5.2--5.3 of that
paper.  In particular, ``m`` and ``n`` are coprime integers greater than one
and ``b = (b_1, ..., b_m)`` is a character orbit in ``(Z/nZ)^m`` satisfying
``sum(b_i) = 0``.

The main formulas implemented here are

.. math::

    \Theta_{\boldsymbol b}(a)
      = (I-P_a) + P_a\left(
          t^{-n}\frac{I-X^m}{I-X}X
        \right)P_a

from equation (14), and

.. math::

    \Psi_{\boldsymbol b}(a)
      = t^{n(1-m)/2}(1-t^n)^{m-2}\delta_m(t)
        (I-X^{-s})(I-M^{-1})^{-1}(I-Y^{-r})

from equation (19).  Although the paper retains ``a`` in the notation for
``Psi``, its displayed rational matrix depends only on ``m``, ``n`` and the
orbit ``b``.  The parameter ``a`` instead specifies the local ring and the
hypothesis under which all entries define germs there.

Sage does not provide the analytic local ring of germs ``O(a)`` used in the
paper as a convenient exact computational parent.  We therefore construct
all matrices in the rational-function field ``Q(zeta_n)(t)``.  Exact
valuation checks at ``t = zeta_n^a`` distinguish elements that are regular or
units in the relevant local ring.  This preserves the algebra needed for the
later signature-jump calculation without introducing floating-point roots.

When ``m`` is even, the paper's factor

.. math::

    \delta_m(t) = (t^{1/2}-t^{-1/2})^{-1}

appears to require a square root of ``t``.  Coprimality then forces ``n`` to
be odd, and the *combined* factor in equation (19) is the ordinary rational
function

.. math::

    \frac{t^{(n(1-m)+1)/2}}{t-1}.

The implementation always uses this combined expression.  Consequently all
returned entries live in ``Q(zeta_n)(t)`` even in the even-``m`` case.
"""

from dataclasses import dataclass, field

from sage.all import (
    CyclotomicField,
    Infinity,
    Integer,
    PolynomialRing,
    gcd,
    identity_matrix,
    matrix,
)


def _is_integer(value):
    """Return whether ``value`` is an accepted exact integer input.

    Python regards booleans as integers, but accepting ``True`` for a torus
    parameter or character coordinate almost certainly hides a caller error.
    As in the other public invariant APIs, booleans are therefore rejected
    explicitly.
    """
    return not isinstance(value, bool) and isinstance(value, (int, Integer))


def _validated_integer(value, name):
    """Return ``value`` as a Sage integer, or raise an informative error."""
    if not _is_integer(value):
        raise TypeError(f"{name} must be an integer.")
    return Integer(value)


def canonical_bezout_coefficients(m, n):
    r"""Return Yanagida's canonical pair ``(r, s)``.

    For coprime ``m,n > 1`` there is a unique solution of

    ``m*r + n*s = 1`` with ``-n < r < 0 < s < m``.

    The negative representative for ``r`` is selected deliberately: powers
    of ``X`` and ``Y`` in the definitions of ``M`` and ``Psi`` depend on this
    convention, not merely on an arbitrary Bezout identity.
    """
    m = _validated_integer(m, "m")
    n = _validated_integer(n, "n")

    if m <= 1 or n <= 1:
        raise ValueError("m and n must both be greater than one.")
    if gcd(m, n) != 1:
        raise ValueError("m and n must be coprime.")

    # ``inverse_mod`` returns the representative in {1, ..., n-1}.  Moving
    # it down by n gives precisely the representative in (-n, 0).
    r = Integer(m.inverse_mod(n) - n)
    s = Integer((1 - m * r) // n)

    # These checks document the normalization and guard against a future
    # change to the construction above.
    if m * r + n * s != 1 or not (-n < r < 0 < s < m):
        raise ArithmeticError("Failed to construct canonical Bezout coefficients.")
    return r, s


def _immutable(matrix_value):
    """Mark a Sage matrix immutable and return it.

    The matrix objects are stored inside frozen dataclasses.  Freezing the
    dataclass prevents attribute reassignment but would not, by itself,
    prevent mutation of individual matrix entries.  Sage's own immutability
    flag closes that remaining hole.
    """
    matrix_value.set_immutable()
    return matrix_value


def _matrix_geometric_sum(base, length):
    r"""Return ``I + base + ... + base^(length-1)``.

    This is the matrix polynomial denoted ``(I-X^m)/(I-X)`` in the paper.
    Computing the finite sum is essential: ``I-X`` is singular at the roots
    of interest, whereas the quotient notation denotes a polynomial identity
    and never asks us to invert ``I-X``.
    """
    length = _validated_integer(length, "geometric-sum length")
    if length <= 0:
        raise ValueError("geometric-sum length must be positive.")

    # Construct two independent identities so the sum does not depend on
    # whether a future Sage matrix backend implements ``*=`` by mutation or
    # by rebinding.
    result = identity_matrix(base.base_ring(), base.nrows())
    power = identity_matrix(base.base_ring(), base.nrows())
    for _ in range(1, length):
        power *= base
        result += power
    return result


def _combined_delta_prefactor(t, m, n, positive_exponent=False):
    r"""Encode a power of ``t`` times Yanagida's ``delta_m`` in ``K(t)``.

    With ``positive_exponent=False`` this returns
    ``t^(n(1-m)/2) * delta_m(t)``, the factor occurring in equations
    (17) and (19).  With ``positive_exponent=True`` it returns
    ``t^(n(m-1)/2) * delta_m(t)``, the representative displayed in the
    target of Theorem 1.3.

    The two branches below eliminate all half powers algebraically before
    constructing the rational function.
    """
    signed_exponent_numerator = n * (m - 1)
    if not positive_exponent:
        signed_exponent_numerator *= -1

    if m % 2 == 1:
        # Here m-1 is even, so division by two is integral.
        return t ** (signed_exponent_numerator // 2)

    # If m is even and gcd(m,n)=1, n is odd.  Consequently the exponent in
    # the following expression is an integer:
    #
    # t^(k/2)/(t^(1/2)-t^(-1/2)) = t^((k+1)/2)/(t-1).
    return t ** ((signed_exponent_numerator + 1) // 2) / (t - 1)


def yanagida_involution(value):
    r"""Apply the involution ``f(t) -> conjugate(f(t^-1))`` exactly.

    ``value`` must belong to a univariate rational-function field whose
    coefficient field supports complex conjugation, as cyclotomic fields do.
    Numerator and denominator polynomials are expanded coefficient by
    coefficient so the substitution ``t -> t^-1`` never relies on symbolic
    expressions or numerical approximations.
    """
    function_field = value.parent()
    t = function_field.gen()

    def transform_polynomial(polynomial):
        transformed = function_field.zero()
        for exponent, coefficient in enumerate(polynomial.list()):
            transformed += coefficient.conjugate() * t ** (-exponent)
        return transformed

    return (
        transform_polynomial(value.numerator())
        / transform_polynomial(value.denominator())
    )


def yanagida_adjoint(matrix_value):
    r"""Return ``matrix_value^(# transpose)`` for Yanagida's involution."""
    function_field = matrix_value.base_ring()
    result = matrix(
        function_field,
        matrix_value.ncols(),
        matrix_value.nrows(),
        lambda row, column: yanagida_involution(matrix_value[column, row]),
    )
    return _immutable(result)


def _polynomial_order_at_root(polynomial, root):
    """Return the multiplicity of ``root`` in a nonzero polynomial."""
    if polynomial.is_zero():
        return Infinity

    factor = polynomial.parent().gen() - root
    order = Integer(0)
    quotient = polynomial
    while quotient(root) == 0:
        quotient, remainder = quotient.quo_rem(factor)
        if not remainder.is_zero():
            raise ArithmeticError("Exact root division produced a remainder.")
        order += 1
    return order


def local_valuation(value, root):
    r"""Return the exact ``(t-root)``-valuation of a rational function.

    A nonzero rational function is regular in the local ring of germs at
    ``root`` exactly when this valuation is nonnegative, and is a local unit
    exactly when it is zero.  The zero rational function has valuation
    ``+Infinity``.
    """
    if value.is_zero():
        return Infinity
    return (
        _polynomial_order_at_root(value.numerator(), root)
        - _polynomial_order_at_root(value.denominator(), root)
    )


def is_regular_at(value, root):
    """Return whether ``value`` defines a germ regular at ``t = root``."""
    return local_valuation(value, root) >= 0


def is_local_unit(value, root):
    """Return whether ``value`` is invertible in the local ring at ``root``."""
    return not value.is_zero() and local_valuation(value, root) == 0


@dataclass(frozen=True)
class YanagidaTorusData:
    r"""Exact global matrix data attached to ``T(m,n)`` and an orbit ``b``.

    The constructor validates and normalizes every coordinate modulo ``n``.
    All matrices are built once and marked immutable.  A root-specific model
    is obtained with :meth:`local_model`.

    Attributes named ``C``, ``X``, ``Y`` and ``M`` use exactly the notation
    of Section 5.1 of Yanagida's paper.  ``psi`` is equation (19); it is stored
    globally because its displayed expression does not actually contain the
    localization parameter ``a``.
    """

    m: object
    n: object
    b: tuple
    coefficient_field: object = field(init=False, repr=False)
    polynomial_ring: object = field(init=False, repr=False)
    function_field: object = field(init=False, repr=False)
    t: object = field(init=False, repr=False)
    zeta: object = field(init=False, repr=False)
    r: object = field(init=False)
    s: object = field(init=False)
    C: object = field(init=False, repr=False)
    X: object = field(init=False, repr=False)
    Y: object = field(init=False, repr=False)
    M: object = field(init=False, repr=False)
    psi: object = field(init=False, repr=False)
    symmetric_prefactor: object = field(init=False, repr=False)
    zero_surgery_twisted_alexander_order: object = field(
        init=False,
        repr=False,
    )

    def __post_init__(self):
        m = _validated_integer(self.m, "m")
        n = _validated_integer(self.n, "n")
        r, s = canonical_bezout_coefficients(m, n)

        if not isinstance(self.b, (list, tuple)):
            raise TypeError("b must be a list or tuple with one entry per orbit point.")
        if len(self.b) != m:
            raise ValueError(f"b must contain exactly m={m} entries.")

        normalized_b = []
        for index, coordinate in enumerate(self.b):
            coordinate = _validated_integer(coordinate, f"b[{index}]")
            normalized_b.append(Integer(coordinate % n))
        normalized_b = tuple(normalized_b)

        if sum(normalized_b, Integer(0)) % n != 0:
            raise ValueError("The character orbit must satisfy sum(b_i) = 0 modulo n.")

        coefficient_field = CyclotomicField(n)
        zeta = coefficient_field.gen()
        polynomial_ring = PolynomialRing(coefficient_field, "t")
        function_field = polynomial_ring.fraction_field()
        t = function_field.gen()
        identity = identity_matrix(function_field, m)

        # C is the cyclic shift matrix with a single t in its lower-left
        # corner.  Its defining consistency relation is C^m = t*I.
        C = matrix(function_field, m, m)
        for row in range(m - 1):
            C[row, row + 1] = 1
        C[m - 1, 0] = t
        X = C ** n

        # Y records the character orbit on its diagonal.  The condition on
        # sum(b_i) is precisely what makes the associated representation have
        # the required determinant behavior.
        Y = t * matrix.diagonal(
            function_field,
            [zeta ** coordinate for coordinate in normalized_b],
        )
        M = X ** s * Y ** r

        symmetric_prefactor = _combined_delta_prefactor(t, m, n)
        psi = (
            symmetric_prefactor
            * (1 - t ** n) ** (m - 2)
            * (identity - X ** (-s))
            * (identity - M ** (-1)).inverse()
            * (identity - Y ** (-r))
        )

        # Equation (18) is the order for the representation on zero surgery.
        # The name is intentionally longer than "twisted Alexander
        # polynomial": the existing package function uses the knot-exterior
        # normalization of Proposition 3.3 in a different paper.
        denominator = 1 - t
        for coordinate in normalized_b:
            denominator *= 1 - zeta ** coordinate * t
        zero_surgery_order = (1 - t ** n) ** (m - 1) / denominator

        object.__setattr__(self, "m", m)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "b", normalized_b)
        object.__setattr__(self, "coefficient_field", coefficient_field)
        object.__setattr__(self, "polynomial_ring", polynomial_ring)
        object.__setattr__(self, "function_field", function_field)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "zeta", zeta)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "C", _immutable(C))
        object.__setattr__(self, "X", _immutable(X))
        object.__setattr__(self, "Y", _immutable(Y))
        object.__setattr__(self, "M", _immutable(M))
        object.__setattr__(self, "psi", _immutable(psi))
        object.__setattr__(self, "symmetric_prefactor", symmetric_prefactor)
        object.__setattr__(
            self,
            "zero_surgery_twisted_alexander_order",
            zero_surgery_order,
        )

    def local_model(self, a):
        """Return the exact model localized at ``t = zeta_n^a``."""
        return YanagidaLocalModel(self, a)


@dataclass(frozen=True)
class YanagidaLocalPairing:
    r"""The generic local pairing presentation from Yanagida's Theorem 1.3.

    ``presentation`` is ``Theta_b(a)`` and presents the local twisted
    Alexander module.  ``matrix`` is ``t^n Psi_b(a)``: for column vectors
    ``f`` and ``g``, the numerator of the pairing is
    ``f^(# transpose) * matrix * g``.

    ``order`` is the representative printed in the target of Theorem 1.3.
    ``symmetric_order`` is equation (17), which is invariant under the
    involution.  They differ by ``t^(n(m-1))``, a unit in every local ring
    considered here, and therefore generate the same local ideal.
    """

    a: object
    root: object
    presentation: object = field(repr=False)
    matrix: object = field(repr=False)
    order: object = field(repr=False)
    symmetric_order: object = field(repr=False)


@dataclass(frozen=True)
class YanagidaLocalModel:
    r"""Root-specific local data for equations (14) and (19).

    Equation (14), and hence ``theta``, is valid for every nonzero ``a`` in
    ``Z/nZ``.  The pairing formula from Theorem 1.3 has the additional
    restriction ``a not in {-b_1, ..., -b_m}``.  The :attr:`pairing` property
    enforces that restriction instead of returning a rational matrix that
    does not define the claimed local germ.
    """

    data: YanagidaTorusData
    a: object
    root: object = field(init=False, repr=False)
    projection: object = field(init=False, repr=False)
    theta: object = field(init=False, repr=False)
    is_generic: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.data, YanagidaTorusData):
            raise TypeError("data must be a YanagidaTorusData object.")

        a = _validated_integer(self.a, "a") % self.data.n
        if a == 0:
            raise ValueError("a must be nonzero modulo n.")

        function_field = self.data.function_field
        identity = identity_matrix(function_field, self.data.m)
        root = self.data.zeta ** a

        # P_a retains the coordinates for which b_i != -a and deletes the
        # exceptional coordinates.  Notice that this convention makes P=I
        # exactly in the generic case covered by Theorem 1.3.
        diagonal = [
            Integer(coordinate != (-a) % self.data.n)
            for coordinate in self.data.b
        ]
        projection = matrix.diagonal(function_field, diagonal)

        geometric_sum = _matrix_geometric_sum(self.data.X, self.data.m)
        theta = (
            (identity - projection)
            + projection
            * (
                self.data.t ** (-self.data.n)
                * geometric_sum
                * self.data.X
            )
            * projection
        )

        object.__setattr__(self, "a", Integer(a))
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "projection", _immutable(projection))
        object.__setattr__(self, "theta", _immutable(theta))
        object.__setattr__(self, "is_generic", all(diagonal))

    @property
    def pairing(self):
        """Return the Theorem 1.3 pairing, rejecting exceptional roots."""
        if not self.is_generic:
            excluded = sorted(
                {
                    (-coordinate) % self.data.n
                    for coordinate in self.data.b
                    if coordinate % self.data.n != 0
                }
            )
            raise ValueError(
                "Yanagida's pairing formula requires a not congruent to any "
                f"-b_i modulo n; excluded nonzero values are {excluded}."
            )

        # The theorem guarantees regularity, but checking it here protects the
        # public object from accidental formula or normalization regressions.
        for entry in self.data.psi.list():
            if not is_regular_at(entry, self.root):
                raise ArithmeticError(
                    "Psi contains an entry with a pole at the localization root."
                )

        t = self.data.t
        m = self.data.m
        n = self.data.n
        pairing_matrix = _immutable(t ** n * self.data.psi)
        theorem_order = (
            _combined_delta_prefactor(t, m, n, positive_exponent=True)
            * (1 - t ** n) ** (m - 1)
        )
        symmetric_order = (
            self.data.symmetric_prefactor * (1 - t ** n) ** (m - 1)
        )

        return YanagidaLocalPairing(
            a=self.a,
            root=self.root,
            presentation=self.theta,
            matrix=pairing_matrix,
            order=theorem_order,
            symmetric_order=symmetric_order,
        )
