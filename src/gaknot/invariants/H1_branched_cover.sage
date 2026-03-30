from sage.all import Integer, ZZ, PolynomialRing, matrix, gcd

class BranchedCoverHomology:
    """
    Represents the first homology group of an N-fold branched cover of a knot.
    
    The group structure mirrors the knot's hierarchical decomposition:
    1. Top-Level: Connected sum components (Summands).
    2. Mid-Level: Satellite layers within each summand (Companion -> Pattern).
    3. Low-Level: Individual generators and their moduli.
    """
    def __init__(self, knot, cover_degree, decomposition=None):
        """
        Args:
            knot: The GeneralizedAlgebraicKnot object.
            cover_degree: The degree of the cover (N).
            decomposition: (Optional) A pre-computed list of homology components 
                           to bypass calculation (used in __add__).
        """
        if not isinstance(cover_degree, (int, Integer)):
            raise TypeError(f'The cover degree should be of type `int` or `Integer`. Got {type(cover_degree)}.')

        if cover_degree < 2:
            raise ValueError(f'The cover degree must be at least two. Got N = {cover_degree}.')
        
        self._cover_degree = cover_degree

        # Use type name checking to bypass Jupyter reload conflicts
        if type(knot).__name__ != 'GeneralizedAlgebraicKnot':
            raise TypeError(f'The knot argument must be of type GeneralizedAlgebraicKnot. Got {type(knot)}.')

        self._knot = knot

        # self._decomposition is a list of dictionaries.
        # Each dict represents one connected sum component of the knot:
        # {
        #    'index': int,              # Index in the knot.description
        #    'sign': int,               # The sign (+1 or -1)
        #    'description': list,       # The [(p,q), (r,s)...] cable description
        #    'layers': list             # List of dicts representing satellite stages (Outer -> Inner)
        # }
        if decomposition is not None:
            self._decomposition = decomposition
        else:
            self._decomposition = self._compute_homology()

    def _compute_homology(self):
        """Computes the homology for each summand of the knot independently."""
        decomposition = []
        
        # We iterate over the knot's description to maintain 1-to-1 correspondence
        for i, (sign, cable_desc) in enumerate(self._knot.description):
            
            # Compute the deep structure (layers) for this component
            # Note: Even basic torus knots are treated as a 1-layer iterated knot
            layers = self._from_iterated_torus_knot(cable_desc, self._cover_degree)
            
            component_data = {
                'index': i,
                'sign': sign,
                'description': cable_desc,
                'layers': layers
            }
            decomposition.append(component_data)
            
        return decomposition

    @staticmethod
    def _from_torus_knot(p, q, N):
        """Helper to compute invariant factors for T(p,q)."""
        from gaknot.utils.utility import alexander_polynomial_torus_knot
        if N == 1:
            return []
            
        Delta = alexander_polynomial_torus_knot(p, q)
        d = Delta.degree()
        coeffs = Delta.list()
        
        C = matrix(ZZ, d, d)
        for i in range(d - 1):
            C[i + 1, i] = 1
        for i in range(d):
            C[i, d - 1] = -coeffs[i]
            
        N_mod = N % (p * q)
        I = matrix.identity(ZZ, d)
        M = (C**N_mod) - I
        
        D = M.smith_form()[0]
        # We include 1s here; filtering happens at the layer construction level
        invariant_factors = [D[i, i] for i in range(d)]
        
        return invariant_factors

    @classmethod
    def _from_iterated_torus_knot(cls, cable_sequence, N):
        """
        Helper to compute homology structure using Litherland's satellite formula.
        Returns a list of 'Layer' dictionaries.
        """
        layers = []
        current_N = N
        multiplier = 1
        
        # Traverse from the outermost pattern down to the innermost companion
        # cable_sequence indices: 0 (inner) -> len-1 (outer)
        # reversed iterator: Outer -> Inner
        
        # We iterate by index explicitly to track the cable_index
        for i in range(len(cable_sequence) - 1, -1, -1):
            p, q = cable_sequence[i]
            
            # Compute the pattern's contribution (homology of T(p,q) cover)
            base_factors = cls._from_torus_knot(p, q, current_N)
            
            # Filter trivial factors (1s) and sort
            cleaned_factors = sorted([f for f in base_factors if f != 1])
            
            layer_data = {
                'cable_index': i,         # Index in the original [(p,q),...] list
                'parameters': (p, q),
                'effective_N': current_N, # The N used for this specific shell
                'multiplicity': multiplier, # Number of copies of this group
                'base_factors': cleaned_factors
            }
            layers.append(layer_data)
                
            # Update the cover degree and multiplier for the next inner companion
            d = gcd(current_N, p)
            current_N = current_N // d
            multiplier = multiplier * d
            
        return layers
    
    def __add__(self, other):
        """Computes the direct sum of two homology groups."""
        if type(self).__name__ != type(other).__name__:
            raise TypeError("Can only add another BranchedCoverHomology object.")
        if self.cover_degree != other.cover_degree:
            raise ValueError(f"Cannot add homologies of different cover degrees: {self.cover_degree} and {other.cover_degree}.")
            
        new_knot = self.knot + other.knot
        
        # When adding, we concatenate the decomposition lists (deep copy structure)
        len_self = len(self._decomposition)
        new_decomposition = []
        
        # Copy self (using simple list slice/copy for the top level, layers are dicts)
        # We rely on the fact that we won't mutate the inner dicts later
        for comp in self._decomposition:
            new_decomposition.append(comp.copy())
            
        # Copy other (adjusting component indices)
        for comp in other._decomposition:
            new_comp = comp.copy()
            new_comp['index'] += len_self
            new_decomposition.append(new_comp)
        
        return type(self)(
            new_knot,
            self.cover_degree, 
            decomposition=new_decomposition
        )

    # --- Accessors ---

    def __getitem__(self, i):
        """Returns the structural dictionary for the i-th connected sum component."""
        if int(i) < 0 or int(i) >= len(self._decomposition):
            raise IndexError("Summand index out of range.")
        return self._decomposition[i]

    def __len__(self):
        """Returns the number of connected sum components."""
        return len(self._decomposition)

    @property
    def knot(self):
        return self._knot
    
    @property
    def cover_degree(self):
        return self._cover_degree

    @property
    def invariant_factors(self):
        """
        Returns the flattened list of all invariant factors for backward compatibility.
        """
        all_factors = []
        for comp in self._decomposition:
            all_factors.extend(self._get_component_factors(comp))
        return sorted(all_factors)
    
    @property
    def all_invariant_factors(self):
        """
        Returns all invariant factors in structural order (Summands -> Layers -> Multiplicity -> Factors).
        Unlike `invariant_factors`, this is NOT sorted and reflects the internal hierarchy.
        """
        factors = []
        for comp in self._decomposition:
            for layer in comp['layers']:
                # Each layer contributes 'multiplicity' copies of 'base_factors'
                for _ in range(layer['multiplicity']):
                    factors.extend(layer['base_factors'])
        return factors
    
    def zero(self):
        """Returns the zero element of the homology group."""
        return BranchedCoverHomologyElement(self, [0] * len(self.all_invariant_factors))

    def element(self, values):
        """Creates an element of the homology group from a list of values."""
        return BranchedCoverHomologyElement(self, values)

    @staticmethod
    def _get_component_factors(component_data):
        """Helper to flatten the factors of a single connected sum component from its layers."""
        factors = []
        for layer in component_data['layers']:
            # Each layer contributes 'multiplicity' copies of 'base_factors'
            for _ in range(layer['multiplicity']):
                factors.extend(layer['base_factors'])
        return factors

    @property
    def decomposition(self):
        """Returns the full structural breakdown of the homology."""
        return self._decomposition

    @property
    def betti_number(self):
        """Returns the rank of the free abelian part of the homology."""
        return self.invariant_factors.count(0)

    def __str__(self):
        """Detailed string representation showing the splitting."""
        if not self.invariant_factors:
            return "0"
        
        parts = []
        for comp in self._decomposition:
            # Flatten this component's factors to display the group summary
            factors = self._get_component_factors(comp)
            
            if not factors:
                continue
                
            group_str = " \u2295 ".join([f"Z/{f}Z" if f != 0 else "Z" for f in factors])
            
            # Create a label for the knot part (e.g. "T(2,3)")
            desc_str = "T(" + "; ".join([f"{p},{q}" for p, q in comp['description']]) + ")"
            sign_str = "-" if comp['sign'] == -1 else ""
            
            parts.append(f"({group_str})[{sign_str}{desc_str}]")
            
        return " \u2295 ".join(parts)
        
    def __repr__(self):
        return f"BranchedCoverHomology(knot='{self.knot}', N={self.cover_degree})"


class BranchedCoverHomologyElement:
    """
    Represents an element of the first homology group of a branched cover of a knot.
    
    Elements are represented as a flat list of integers corresponding to the
    structural invariant factors of the homology group.
    """
    def __init__(self, homology, values):
        """
        Args:
            homology: A BranchedCoverHomology object.
            values: A list of integers (flat or nested) matching the homology structure.
        """
        if type(homology).__name__ != 'BranchedCoverHomology':
            raise TypeError(f"Expected a BranchedCoverHomology object, got {type(homology)}.")
            
        self._homology = homology
        factors = homology.all_invariant_factors
        
        # Flatten and validate values
        if isinstance(values, (int, Integer)):
             if len(factors) == 1:
                 flat_values = [values]
             else:
                 raise ValueError(f"Single value provided but homology has {len(factors)} generators.")
        elif isinstance(values, (list, tuple)):
            # Check if it's already flat and matches length
            if len(values) == len(factors) and all(not isinstance(v, (list, tuple)) for v in values):
                flat_values = values
            else:
                # Attempt to flatten nested structure (Summands -> Layers -> Values)
                flat_values = self._flatten_nested_values(values, homology)
        else:
            raise TypeError(f"Values must be an integer or a list of integers. Got {type(values)}.")

        if len(flat_values) != len(factors):
            raise ValueError(f"Value mismatch: Homology has {len(factors)} generators, but {len(flat_values)} values were provided.")
            
        # Store values reduced modulo the invariant factors
        self._values = [Integer(v) % f if f != 0 else Integer(v) for v, f in zip(flat_values, factors)]

    @staticmethod
    def _flatten_nested_values(nested_values, homology):
        """Helper to flatten nested values following the hierarchy of the knot."""
        if len(nested_values) != len(homology.decomposition):
            raise ValueError(f"Summand count mismatch: Expected {len(homology.decomposition)}, got {len(nested_values)}.")
            
        flat = []
        for c_idx, (comp_data, comp_values) in enumerate(zip(homology.decomposition, nested_values)):
            layers = comp_data['layers']
            if len(comp_values) != len(layers):
                raise ValueError(f"Layer count mismatch in Summand {c_idx}: Expected {len(layers)}, got {len(comp_values)}.")
                
            for l_idx, (layer, layer_values) in enumerate(zip(layers, comp_values)):
                multiplicity = layer['multiplicity']
                base_factors = layer['base_factors']
                expected_count = multiplicity * len(base_factors)
                
                if len(layer_values) != expected_count:
                    raise ValueError(f"Value mismatch in Summand {c_idx}, Layer {l_idx}: Expected {expected_count} values, got {len(layer_values)}.")
                
                flat.extend(layer_values)
        return flat

    @property
    def homology(self):
        return self._homology

    @property
    def values(self):
        """Returns the flattened list of values."""
        return self._values

    def __add__(self, other):
        if not isinstance(other, BranchedCoverHomologyElement):
            raise TypeError("Can only add another BranchedCoverHomologyElement.")
        if self._homology != other._homology:
            raise ValueError("Cannot add elements from different homology groups.")
            
        new_values = [v1 + v2 for v1, v2 in zip(self._values, other._values)]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __sub__(self, other):
        if not isinstance(other, BranchedCoverHomologyElement):
            raise TypeError("Can only subtract another BranchedCoverHomologyElement.")
        if self._homology != other._homology:
            raise ValueError("Cannot subtract elements from different homology groups.")
            
        new_values = [v1 - v2 for v1, v2 in zip(self._values, other._values)]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __neg__(self):
        new_values = [-v for v in self._values]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, Integer)):
            raise TypeError("Scalar multiplication only supported for integers.")
        new_values = [v * scalar for v in self._values]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __eq__(self, other):
        if not isinstance(other, BranchedCoverHomologyElement):
            return False
        return self._homology == other._homology and self._values == other._values

    def __repr__(self):
        return f"BranchedCoverHomologyElement(homology={repr(self._homology)}, values={self._values})"

    def __str__(self):
        return str(self._values)
