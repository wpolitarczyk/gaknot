from gaknot.invariants.ckp import ckp_torus_knot_data


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
    # Build the Proposition 3.2 matrices first and obtain the polynomial as
    # the quotient of their two Fox determinants.  This keeps the historical
    # public function while ensuring that its result and the auditable CKP
    # representation data cannot drift into different conventions.
    return ckp_torus_knot_data(knot, character).exterior_twisted_alexander
