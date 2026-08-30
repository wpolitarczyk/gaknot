from sage.all import CyclotomicField, PolynomialRing

from gaknot.invariants.torus_character import torus_character_orbit


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

    # The same orbit is used by the twisted Alexander formula, Yanagida's
    # pairing matrices, and the satellite phase shifts.  Delegate the basis
    # conversion to one shared implementation so all three calculations use
    # exactly the same cyclic ordering.  A one-layer torus knot has one copy of
    # its outer layer, hence the two indices below are structurally forced.
    generator_values = character.restrict_to_layer(0, 0)[0]
    orbit = torus_character_orbit(p, q, generator_values)

    # Proposition 3.3 lives in the rational-function field Q(zeta_q)(t).
    K = CyclotomicField(q)
    zeta = K.gen()
    R = PolynomialRing(K, 't')
    t = R.gen()

    # Assemble (1 - t^q)^(p-1) / product_j(t*zeta_q^(a_j) - 1).
    numerator = (1 - t ** q) ** (p - 1)
    denominator = R(1)
    for a in orbit.a_values:
        denominator *= (t * zeta ** a - 1)

    # Preserve the representative selected by the displayed formula.  In
    # particular, do not multiply by a power of t or a nonzero field element,
    # even though either operation represents the same abstract invariant.
    return numerator / denominator
