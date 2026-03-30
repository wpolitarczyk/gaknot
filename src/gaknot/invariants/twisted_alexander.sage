from sage.all import ZZ, CyclotomicField, PolynomialRing, matrix, Integer
from gaknot.utils.utility import alexander_polynomial_torus_knot

def twisted_alexander_torus_knot(knot, character):
    """
    Computes the metabelian twisted Alexander polynomial of a torus knot T(p,q)
    associated to a character chi on the p-fold branched cover.
    
    Formula from Conway-Kim-Politarczyk, Proposition 3.3:
    Delta(t) = (1 - t^q)^(p-1) / prod_{j=0}^{p-1} (t * zeta_q^{a_j} - 1)
    
    Args:
        knot: A GeneralizedAlgebraicKnot object representing T(p,q).
        character: A Character object defined on the p-fold branched cover of the knot.
        
    Returns:
        A rational function in the ring of polynomials over a cyclotomic field.
    """
    if not knot.is_positive_torus_knot():
        raise ValueError("Knot must be a positive torus knot T(p,q).")
    
    if character is None or type(character).__name__ != 'Character':
        raise TypeError(f"Expected a Character object, got {type(character)}.")

    sign, cable_desc = knot.description[0]
    if sign == -1:
        raise NotImplementedError("Twisted Alexander polynomial for negative torus knots not yet implemented.")
        
    p, q = cable_desc[0]
    h1 = character.homology
    
    if h1.cover_degree != p:
        raise ValueError(f"Formula requires character on the {p}-fold cover (winding number of the torus knot). Got N={h1.cover_degree}, expected p={p}.")
        
    # Alexander polynomial and companion matrix of T(p,q)
    Delta = alexander_polynomial_torus_knot(p, q)
    d = Delta.degree()
    coeffs = Delta.list()
    
    C = matrix(ZZ, d, d)
    for i in range(d - 1):
        C[i + 1, i] = 1
    for i in range(d):
        C[i, d - 1] = -coeffs[i]
        
    # The homology H1(Sigma_p(T(p,q))) is isomorphic to ZZ^d / (C^p - I) ZZ^d
    M = (C**p) - matrix.identity(ZZ, d)
    D, U, V = M.smith_form() # U*M*V = D
    
    # Identify indices of non-trivial invariant factors (moduli)
    diag = D.diagonal()
    idx_factors = [i for i, f in enumerate(diag) if f != 1]
    
    # The generators x_j are given by the orbit of the image of the first basis vector e0
    # under the deck transformation t (acting via C).
    a_values = []
    v = matrix(ZZ, d, 1)
    v[0, 0] = 1 # e0
    
    for _ in range(p):
        # Coordinates of the current element in the Smith basis
        # The projection maps v to U*v in the basis where M is diagonal
        coords = U * v
        
        # Extract values for the generators that actually contribute to the homology
        # (those with modulus > 1).
        gen_values = [coords[i, 0] for i in idx_factors]
        
        # Create a homology element and evaluate the character
        # Note: Nested structure [[values]] for 1 summand and 1 layer
        el = h1.element([[gen_values]])
        val = character(el)
        
        # a_j = q * chi(x_j) is an integer
        a_values.append(Integer(val * q))
        
        # Move to the next element in the orbit: x_{j+1} = t * x_j
        v = C * v
        
    # Construct the polynomial/rational function in Q(zeta_q)[t]
    K = CyclotomicField(q)
    zeta = K.gen()
    R = PolynomialRing(K, 't')
    t = R.gen()
    
    # Numerator: (1 - t^q)^(p-1)
    num = (1 - t**q)**(p - 1)
    
    # Denominator: prod_{j=0}^{p-1} (t * zeta_q^{a_j} - 1)
    den = R(1)
    for a in a_values:
        den *= (t * zeta**a - 1)
        
    return num / den
