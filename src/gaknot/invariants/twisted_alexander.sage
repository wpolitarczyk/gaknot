from sage.all import ZZ, CyclotomicField, PolynomialRing, matrix, Integer
from gaknot.utils.utility import alexander_polynomial_torus_knot


def twisted_alexander_torus_knot(knot, character):
    """
    Compute the metabelian twisted Alexander polynomial of T(p,q).

    The character chi is defined on H_1(Sigma_p(T(p,q))) and takes values in
    Q/Z.  For this cover the homology is (Z/qZ)^(p-1), so chi may equivalently
    be regarded as a Z/qZ-valued character.

    Twisted Alexander polynomials are intrinsically defined only up to
    multiplication by units in the Laurent polynomial ring.  This function
    fixes that ambiguity by returning exactly the representative displayed in
    Proposition 3.3, with numerator (1 - t^q)^(p-1) and the ordered product in
    the denominator.  It performs no additional rescaling by a unit.
    
    Formula from Conway-Kim-Politarczyk, Proposition 3.3:
    Delta(t) = (1 - t^q)^(p-1) / prod_{j=0}^{p-1} (t * zeta_q^{a_j} - 1)
    
    Args:
        knot: A GeneralizedAlgebraicKnot object representing T(p,q).
        character: A Character object defined on the p-fold branched cover of
                   the same knot.
        
    Returns:
        The fixed Proposition 3.3 representative as a rational function over
        the q-th cyclotomic field.
    """
    # A negative torus knot is still a single torus knot, but the orientation
    # convention needed for its formula has not yet been implemented.  Handle
    # it before the broader domain check so callers receive the specific error
    # instead of the generic "must be positive" rejection.
    if knot.is_negative_torus_knot():
        raise NotImplementedError(
            "Twisted Alexander polynomial for negative torus knots not yet implemented."
        )

    if not knot.is_positive_torus_knot():
        raise ValueError("Knot must be a positive torus knot T(p,q).")
    
    if character is None or type(character).__name__ != 'Character':
        raise TypeError(f"Expected a Character object, got {type(character)}.")

    _, cable_desc = knot.description[0]
    p, q = cable_desc[0]
    h1 = character.homology

    # The orbit elements used below come from the p-fold branched cover of
    # this particular torus knot.  A character on another knot has a different
    # Smith basis, even when its cover degree and number of generators happen
    # to agree.  Compare descriptions rather than object identity so that a
    # separately constructed copy of the same knot remains valid.
    if h1.knot.description != knot.description:
        raise ValueError(
            "Character must be defined on the homology of the supplied knot."
        )
    
    if h1.cover_degree != p:
        raise ValueError(
            f"Formula requires character on the {p}-fold cover "
            f"(winding number of the torus knot). Got N={h1.cover_degree}, "
            f"expected p={p}."
        )

    # The companion matrix C presents multiplication by the Alexander-module
    # variable in the standard basis determined by Delta_{T(p,q)}.  The
    # polynomial is monic, so its coefficients determine the final column.
    Delta = alexander_polynomial_torus_knot(p, q)
    d = Delta.degree()
    coeffs = Delta.list()

    C = matrix(ZZ, d, d)
    for i in range(d - 1):
        C[i + 1, i] = 1
    for i in range(d):
        C[i, d - 1] = -coeffs[i]

    # Passing to the p-fold branched cover imposes C^p = I.  Thus its first
    # homology is the cokernel of M = C^p - I.  Smith form supplies matrices
    # with U*M*V = D, and U converts a vector from the companion basis to the
    # diagonal presentation used by BranchedCoverHomologyElement.
    M = (C ** p) - matrix.identity(ZZ, d)
    D, U, _ = M.smith_form()

    # A diagonal entry 1 presents a trivial cyclic summand and has no
    # coordinate in the public homology representation.  Retain precisely the
    # Smith coordinates belonging to nontrivial summands.
    diag = D.diagonal()
    idx_factors = [i for i, f in enumerate(diag) if f != 1]

    # In the notation of Proposition 3.3, x_0 is represented by the first
    # companion-basis vector and x_j = C^j*x_0.  Evaluating chi around this
    # p-element deck-transformation orbit produces a_0, ..., a_{p-1} in Z/qZ.
    a_values = []
    orbit_vector = matrix(ZZ, d, 1)
    orbit_vector[0, 0] = 1

    for _ in range(p):
        # Project the current x_j to the Smith basis and discard coordinates
        # belonging to the trivial diagonal factors.
        coords = U * orbit_vector
        gen_values = [coords[i, 0] for i in idx_factors]

        # A basic torus knot has one connected-sum component and one layer, so
        # [[gen_values]] is the nested form of this homology element.  Since
        # chi(x_j) is q-torsion, q*chi(x_j) is the integer representative a_j.
        element = h1.element([[gen_values]])
        character_value = character(element)
        a_values.append(Integer(character_value * q))

        # Advance once around the orbit: x_{j+1} = C*x_j.
        orbit_vector = C * orbit_vector

    # Proposition 3.3 lives in the rational-function field Q(zeta_q)(t).
    K = CyclotomicField(q)
    zeta = K.gen()
    R = PolynomialRing(K, 't')
    t = R.gen()

    # Assemble (1 - t^q)^(p-1) / product_j(t*zeta_q^(a_j) - 1).
    numerator = (1 - t ** q) ** (p - 1)
    denominator = R(1)
    for a in a_values:
        denominator *= (t * zeta ** a - 1)

    # Preserve the representative selected by the displayed formula.  In
    # particular, do not multiply by a power of t or a nonzero field element,
    # even though either operation represents the same abstract invariant.
    return numerator / denominator
