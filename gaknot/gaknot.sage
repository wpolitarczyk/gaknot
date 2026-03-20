from sage.all import Integer, gcd

class GeneralizedAlgebraicKnot:
    r"""
    Represents a generalized algebraic knot, formed by taking connected sums 
    of iterated positive cables of positive torus knots and their concordance inverses.

    The knot is described using a nested list/tuple structure:
    `[(sign_1, knot_desc_1), (sign_2, knot_desc_2), ...]`

    1. Top-Level: Connected Sum
       Each element is a pair `(sign, knot_desc)` representing a summand.
       - `sign`: 1 for the knot, -1 for its orientation-reversed mirror image (concordance inverse).
       - `knot_desc`: Description of an iterated torus knot (satellite knot).

    2. Inner Level: Iterated Torus Knot
       Each `knot_desc` is a list of cabling parameters `(p, q)`:
       `[(p_1, q_1), (p_2, q_2), ..., (p_n, q_n)]`
       - Parameters are ordered from the innermost pattern to the outermost companion.
       - Each pair `(p, q)` represents a torus knot $T(p,q)$ and must satisfy $p, q > 1$ 
         and $\gcd(p, q) = 1$.
    """

    def __init__(self, desc):
        """
        Initializes the knot with a validated description.
        """
        # Validate the description before storing it
        self.__class__.verify_description(desc)
        self._desc = desc


    @staticmethod
    def verify_description(desc):
        """
        Verifies if the provided description is a valid generalized algebraic knot.
        Raises TypeError or ValueError with detailed context if the description is invalid.
        Returns True if the description is perfectly valid.
        """
        if not isinstance(desc, (list, tuple)):
            raise TypeError(f"The description must be a list or tuple. Got {type(desc)}.")

        for i, element in enumerate(desc):
            # 1. Check the top-level pair (sign, knot_description)
            if not isinstance(element, (list, tuple)) or len(element) != 2:
                raise ValueError(f"Element at index {i} must be a pair (sign, knot_description).")
            
            sign, knot_desc = element

            # 2. Check the sign
            if sign not in (1, -1):
                raise ValueError(f"Sign at index {i} must be 1 or -1. Got {sign}.")

            # 3. Check the iterated torus knot description container
            if not isinstance(knot_desc, (list, tuple)):
                raise TypeError(f"Knot description at index {i} must be a list or tuple.")

            # 4. Check each cabling step (p, q)
            for j, cable_pair in enumerate(knot_desc):
                if not isinstance(cable_pair, (list, tuple)) or len(cable_pair) != 2:
                    raise ValueError(f"Cable parameter at index {i}, sub-index {j} must be a pair (p, q).")
                
                p, q = cable_pair
                
                # Type check for integers
                if not isinstance(p, (int, Integer)) or not isinstance(q, (int, Integer)):
                    raise TypeError(f"Parameters p and q must be integers. Got {type(p)}, {type(q)} at index {i}, {j}.")
                    
                # Positivity check
                if p <= 1 or q <= 1:
                    raise ValueError(f"Parameters p and q must be > 1. Got ({p}, {q}) at index {i}, {j}.")

                # Coprimality check
                if gcd(p, q) != 1:
                    raise ValueError(f"Parameters p and q must be relatively prime. Got gcd({p}, {q}) != 1 at index {i}, {j}.")
        
        return True


    @property
    def description(self):
        """Read-only access to the knot description."""
        return self._desc


    def signature(self):
        """
        Computes the Levine-Tristram signature function of this generalized algebraic knot.
        Returns a SignatureFunction object.
        """
        # Local import to prevent circular dependencies during package initialization
        from .LT_signature import LT_signature_generalized_algebraic_knot
        
        return LT_signature_generalized_algebraic_knot(self.description)


    def alexander_polynomial(self):
        """
        Computes the Alexander polynomial of the generalized algebraic knot.
        The polynomial of a connected sum is the product of the polynomials of its summands.
        """
        from sage.all import PolynomialRing, ZZ
        # Local import of the helper function if it is in another file
        from .utility import alexander_polynomial_iterated_knot
        
        R = PolynomialRing(ZZ, 't')
        total_poly = R(1)
        
        for sign, knot_desc in self.description:
            poly = alexander_polynomial_iterated_knot(knot_desc)
            total_poly *= poly
            
        return total_poly

    def is_positive_torus_knot(self):
        """
        Checks if the knot is exactly a single positive torus knot.
        """
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == 1 and len(knot_desc) == 1

    def is_negative_torus_knot(self):
        """
        Checks if the knot is exactly a single negative torus knot 
        (the concordance inverse of a positive torus knot).
        """
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == -1 and len(knot_desc) == 1

    def is_iterated_torus_knot(self):
        """
        Checks if the knot is exactly a single positive iterated torus knot.
        (Note: Returns True for basic positive torus knots as well).
        """
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == 1 and len(knot_desc) >= 1

    def is_neg_iterated_torus_knot(self):
        """
        Checks if the knot is exactly a single negative iterated torus knot.
        (Note: Returns True for basic negative torus knots as well).
        """
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == -1 and len(knot_desc) >= 1


    def __add__(self, other):
        """
        Connected sum of two generalized algebraic knots.
        """
        if not isinstance(other, GeneralizedAlgebraicKnot):
            raise TypeError("Can only add another GeneralizedAlgebraicKnot.")
        
        new_knot_desc = self.description + other.description
        return GeneralizedAlgebraicKnot(new_knot_desc)

    def __neg__(self):
        """
        Concordance inverse of the knot (flips the sign of every component).
        """
        new_knot_desc = [(-sign, knot_desc) for sign, knot_desc in self.description]
        return GeneralizedAlgebraicKnot(new_knot_desc)

    def __sub__(self, other):
        return self + (-other)


    def __len__(self):
        """Returns the number of summands in the connected sum."""
        return len(self._desc)
    

    def __getitem__(self, i):
        """
        Allows indexing and slicing of the knot's summands.
        Returns a new GeneralizedAlgebraicKnot.
        """
        try:
            # Handle standard integer indexing (e.g., my_knot[0] or my_knot[-1])
            if isinstance(i, (int, Integer)):
                return GeneralizedAlgebraicKnot([self._desc[int(i)]])
            
            # Handle slicing (e.g., my_knot[0:2])
            elif isinstance(i, slice):
                return GeneralizedAlgebraicKnot(self._desc[i])
                
            else:
                raise TypeError(f"Invalid argument type: {type(i)}")
                
        except IndexError:
            raise IndexError(f"Knot index out of range. The knot has {len(self)} summand(s).")
    

    @staticmethod
    def _it_torus_knot_desc_to_txt(desc):
        """
        Helper method to format the iterated torus knot sequence.
        Example: [(2,3), (6,5)] -> 'T(2,3; 6,5)'
        """
        return 'T(' + '; '.join([f"{p},{q}" for p, q in desc]) + ')'

    def __str__(self):
        """
        Returns the human-readable string representation of the generalized algebraic knot.
        Example: 'T(2,3; 2,5; 3,4) # -T(5,2; 3,7)'
        """
        components = []
        for sign, knot_desc in self.description:
            # Map the sign to a minus string or an empty string
            prefix = "-" if sign == -1 else ""
            knot_str = self._it_torus_knot_desc_to_txt(knot_desc)
            components.append(f"{prefix}{knot_str}")
        
        return ' # '.join(components)

    def __repr__(self):
        """
        Returns the developer representation, which shows the exact data structure 
        needed to recreate the object.
        """
        return f"GeneralizedAlgebraicKnot({self.description})"

    
