"""First homology of cyclic branched covers of generalized algebraic knots.

The main class represents

    H_1(Sigma_N(K); Z),

where ``Sigma_N(K)`` is the ``N``-fold cyclic cover of the three-sphere
branched over ``K``.  A generalized algebraic knot is a signed connected sum
of iterated torus knots, so the implementation retains the provenance of each
cyclic factor instead of storing only the abstract isomorphism type.

There are two useful views of the resulting finitely generated abelian group.
The *structural* view follows connected-sum components and satellite layers;
the *canonical* view applies Smith normal form to the entire group and returns
the usual divisibility-ordered invariant factors.  A stored factor ``m > 1``
means ``Z/mZ``, a factor ``0`` means a free copy of ``Z``, and factors equal to
``1`` are omitted because they describe the trivial group.

Cable descriptions and homology layers use opposite directions.  A cable
description lists the innermost companion first and the outermost pattern
last.  The homology calculation follows the satellite formula from the
outside inward, so its layer dictionaries are ordered outermost first.  Group
elements use the corresponding flattened structural order, but may also be
constructed from values nested by component and layer.
"""

from copy import deepcopy

from sage.all import Integer, ZZ, diagonal_matrix, matrix, gcd
from gaknot.core.gaknot import GeneralizedAlgebraicKnot


class BranchedCoverHomology:
    """First homology of an ``N``-fold cyclic branched cover.

    The stored decomposition mirrors the construction of the knot:

    1. The top level consists of connected-sum components.
    2. Each component contains satellite layers in outer-to-inner order.
    3. Each layer records one list of cyclic factors and the number of copies
       of that list contributed by the satellite formula.

    The sign of a knot component does not change its abstract first homology,
    but is retained in the decomposition and string representation so that a
    factor can still be traced back to the signed knot description.
    """

    def __init__(self, knot, cover_degree, decomposition=None):
        """Construct the homology group and its structural decomposition.

        Args:
            knot: The :class:`GeneralizedAlgebraicKnot` defining the branch
                locus.
            cover_degree: A Python or Sage integer ``N >= 2``.
            decomposition: An optional precomputed structural decomposition.
                This internal construction path is used by addition.  The
                value is deep-copied so the new object never owns mutable data
                supplied by its caller.

        Raises:
            TypeError: If the degree or knot has the wrong type.
            ValueError: If the cover degree is smaller than two.
        """
        if isinstance(cover_degree, bool) or not isinstance(cover_degree, (int, Integer)):
            raise TypeError(f'The cover degree should be of type `int` or `Integer`. Got {type(cover_degree)}.')

        if cover_degree < 2:
            raise ValueError(f'The cover degree must be at least two. Got N = {cover_degree}.')
        
        self._cover_degree = cover_degree

        if not isinstance(knot, GeneralizedAlgebraicKnot):
            raise TypeError(f'The knot argument must be of type GeneralizedAlgebraicKnot. Got {type(knot)}.')

        self._knot = knot

        # ``_decomposition`` is a list with one dictionary per connected-sum
        # component.  Its order agrees with ``knot.description``:
        # {
        #     'index': int,          # position in knot.description
        #     'sign': int,           # +1 or -1 component sign
        #     'description': list,   # cable pairs, inner -> outer
        #     'layers': list,        # homology layers, outer -> inner
        # }
        #
        # Each entry of ``layers`` has the schema documented by
        # ``_from_iterated_torus_knot`` below.
        if decomposition is not None:
            # In particular, addition must not make the result share nested
            # ``layers`` or ``base_factors`` lists with either operand.
            self._decomposition = deepcopy(decomposition)
        else:
            self._decomposition = self._compute_homology()


    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _compute_homology(self):
        """Compute every connected-sum component independently.

        First homology takes a connected sum of knots to the direct sum of the
        corresponding cover homologies.  The method therefore preserves a
        one-to-one correspondence between knot summands and component
        dictionaries rather than flattening all factors immediately.
        """
        decomposition = []
        
        # Iterating in description order keeps component indices stable and
        # makes structural factor order predictable to characters and elements.
        for i, (sign, cable_desc) in enumerate(self._knot.description):
            
            # An ordinary torus knot is the one-layer special case of an
            # iterated torus knot, so both use exactly the same representation.
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
        """Return raw Smith factors for the ``N``-fold cover of ``T(p,q)``.

        Let ``Delta`` be the monic Alexander polynomial of the torus knot and
        let ``C`` be its companion matrix.  The standard companion-matrix
        presentation for the branched-cover homology is ``C^N - I``.  Smith
        normal form over ``ZZ`` then describes its cokernel: a nonunit diagonal
        entry gives a finite cyclic factor, a zero gives a free factor, and a
        unit gives no contribution.

        This helper deliberately returns the unit entries as well.  Removing
        them is a structural concern handled when the satellite layer is
        created, ensuring a single cleanup rule for ordinary and iterated
        torus knots.
        """
        from gaknot.utils.utility import alexander_polynomial_torus_knot

        # The one-fold branched cover is the original three-sphere and has
        # trivial first homology.  Recursive satellite calculations can reach
        # this case even though the public constructor requires N >= 2.
        if N == 1:
            return []
            
        Delta = alexander_polynomial_torus_knot(p, q)
        d = Delta.degree()
        coeffs = Delta.list()
        
        # In the power basis, multiplication by a root of the monic polynomial
        # Delta is represented by its companion matrix.  The subdiagonal shifts
        # basis vectors; the last column encodes the relation Delta(C) = 0.
        C = matrix(ZZ, d, d)
        for i in range(d - 1):
            C[i + 1, i] = 1
        for i in range(d):
            C[i, d - 1] = -coeffs[i]
            
        # Every root of the torus-knot Alexander polynomial is a pq-th root of
        # unity.  Hence C^pq = I and only N modulo pq affects C^N - I.  Reducing
        # the exponent avoids needlessly expensive large matrix powers.
        N_mod = N % (p * q)
        I = matrix.identity(ZZ, d)
        M = (C**N_mod) - I
        
        # The diagonal of the Smith form presents coker(M).  Sage records free
        # summands as zeros on that diagonal.
        D = M.smith_form()[0]
        # Keep 1s here; filtering happens at the layer-construction level.
        invariant_factors = [D[i, i] for i in range(d)]
        
        return invariant_factors

    @classmethod
    def _from_iterated_torus_knot(cls, cable_sequence, N):
        """Compute satellite layers using Litherland's decomposition formula.

        For a ``(p,q)``-cable with current cover degree ``N``, put
        ``d = gcd(N,p)``.  The pattern contributes the homology of the
        ``N``-fold cover of ``T(p,q)``, while the companion contribution occurs
        in ``d`` copies and is computed using cover degree ``N/d``.  Repeating
        this step accounts for every layer of an iterated torus knot.

        Args:
            cable_sequence: Cable pairs ordered innermost to outermost.
            N: Cover degree at the outermost layer.

        Returns:
            A list ordered outermost to innermost.  Each layer dictionary has
            the following fields:

            * ``cable_index``: position of the pair in ``cable_sequence``;
            * ``parameters``: its pair ``(p,q)``;
            * ``effective_N``: cover degree used for this pattern;
            * ``multiplicity``: number of copies of its factor list;
            * ``base_factors``: nontrivial Smith factors for one copy.
        """
        layers = []
        # At the outer boundary there is one copy with the original degree.
        current_N = N
        multiplier = 1
        
        # The source sequence runs inner -> outer, so reverse it to follow the
        # satellite formula outer -> inner.  Explicit indices preserve the
        # connection between a homology layer and its original cable pair.
        for i in range(len(cable_sequence) - 1, -1, -1):
            p, q = cable_sequence[i]
            
            # This is one copy of the pattern contribution.  ``multiplier``
            # records how many parallel copies have arisen at outer stages.
            base_factors = cls._from_torus_knot(p, q, current_N)
            
            # Unit Smith entries describe the zero group and are discarded.
            # Zeros are retained because they encode genuine free summands.
            cleaned_factors = sorted([f for f in base_factors if f != 1])
            
            layer_data = {
                'cable_index': i,           # index in the original cable list
                'parameters': (p, q),
                'effective_N': current_N,   # cover degree for this shell
                'multiplicity': multiplier, # copies of these base factors
                'base_factors': cleaned_factors
            }
            layers.append(layer_data)
                
            # Move to the companion term in the satellite formula.  Every copy
            # already present splits into d companion contributions.
            d = gcd(current_N, p)
            current_N = current_N // d
            multiplier = multiplier * d
            
        return layers
    
    def __add__(self, other):
        """Return the direct sum associated with a connected sum of knots.

        Both operands must use the same cover degree.  Their structural
        decompositions are concatenated, with component indices from the
        second operand shifted to follow those from the first.
        """
        if not isinstance(other, BranchedCoverHomology):
            raise TypeError("Can only add another BranchedCoverHomology object.")
        if self.cover_degree != other.cover_degree:
            raise ValueError(f"Cannot add homologies of different cover degrees: {self.cover_degree} and {other.cover_degree}.")
            
        new_knot = self.knot + other.knot
        
        # Start with an independent copy so the result and left operand never
        # share mutable component, layer, or factor lists.
        len_self = len(self._decomposition)
        new_decomposition = deepcopy(self._decomposition)

        # Copy the right operand for the same reason, then translate its local
        # indices into positions in the combined connected sum.
        for new_comp in deepcopy(other._decomposition):
            new_comp['index'] += len_self
            new_decomposition.append(new_comp)
        
        return type(self)(
            new_knot,
            self.cover_degree, 
            decomposition=new_decomposition
        )


    # ------------------------------------------------------------------
    # Structural access and group invariants
    # ------------------------------------------------------------------

    def __getitem__(self, i):
        """Return a deep copy of connected-sum component ``i``.

        Component indices deliberately do not support Python's negative-index
        convention: they are persistent identifiers in the decomposition, not
        merely positions in a transient sequence.
        """
        if not isinstance(i, (int, Integer)) or isinstance(i, bool):
            raise TypeError("Summand index must be an integer.")

        i = int(i)
        if i < 0 or i >= len(self._decomposition):
            raise IndexError("Summand index out of range.")
        return deepcopy(self._decomposition[i])

    def __len__(self):
        """Return the number of connected-sum components, not generators."""
        return len(self._decomposition)

    @property
    def knot(self):
        """Return the generalized algebraic knot defining the cover."""
        return self._knot
    
    @property
    def cover_degree(self):
        """Return the degree ``N`` of the cyclic branched cover."""
        return self._cover_degree

    @property
    def invariant_factors(self):
        """
        Returns the sorted structural cyclic moduli for backward compatibility.

        These values retain the decomposition into knot summands and satellite
        layers. They need not form the canonical invariant-factor decomposition;
        for example, Z/3Z ⊕ Z/5Z is represented here by [3, 5].
        """
        all_factors = []
        for comp in self._decomposition:
            all_factors.extend(self._get_component_factors(comp))
        return sorted(all_factors)

    @property
    def canonical_invariant_factors(self):
        """Return the canonical invariant factors of the complete group.

        Unlike the structural properties, this view forgets which knot and
        satellite layer supplied each generator.  Smith-normalizing the
        diagonal presentation produces factors ``d_1 | ... | d_r``; this can,
        for example, replace structural factors ``[3, 5]`` by ``[15]``.  Unit
        factors are removed, while zero factors representing free summands
        remain.
        """
        factors = self.all_invariant_factors
        if not factors:
            return []

        # The structural direct sum has diagonal presentation matrix
        # diag(factors), so one additional Smith form canonicalizes it.
        smith_form = diagonal_matrix(ZZ, factors).smith_form()[0]
        return [
            smith_form[i, i]
            for i in range(smith_form.nrows())
            if smith_form[i, i] != 1
        ]
    
    @property
    def all_invariant_factors(self):
        """Return cyclic moduli in structural rather than canonical order.

        The flattening order is components, layers, repeated copies of a layer,
        and finally factors within one copy.  This order is shared by homology
        elements and characters.  Unlike ``invariant_factors``, the result is
        not sorted, so it continues to reflect the internal hierarchy.
        """
        factors = []
        for comp in self._decomposition:
            for layer in comp['layers']:
                # A layer contributes ``multiplicity`` consecutive copies of
                # the same one-copy factor list.
                for _ in range(layer['multiplicity']):
                    factors.extend(layer['base_factors'])
        return factors
    
    def zero(self):
        """Return the identity element in flattened structural coordinates.

        One zero is supplied for every structural factor, including free
        factors, so the result is both the additive identity and a torsion
        element even when the ambient homology has positive free rank.
        """
        return BranchedCoverHomologyElement(self, [0] * len(self.all_invariant_factors))

    def element(self, values):
        """Construct and normalize an element belonging to this group.

        ``values`` may be a single integer for a one-generator group, a flat
        list in ``all_invariant_factors`` order, or a nested description with
        shape ``[component][layer][coordinate]``.  The element constructor
        validates the shape, flattens repeated layer copies, reduces finite
        coordinates, and preserves free coordinates as integers.
        """
        return BranchedCoverHomologyElement(self, values)

    @staticmethod
    def _get_component_factors(component_data):
        """Flatten one component's layers, copies, and factors in that order."""
        factors = []
        for layer in component_data['layers']:
            # Keep every repeated copy contiguous within its source layer.
            for _ in range(layer['multiplicity']):
                factors.extend(layer['base_factors'])
        return factors

    @property
    def decomposition(self):
        """Return a deep copy of the full component-and-layer decomposition."""
        return deepcopy(self._decomposition)

    @property
    def betti_number(self):
        """Return the free rank, represented by the number of zero factors."""
        return self.invariant_factors.count(0)

    def __str__(self):
        """Display the structural direct sum, labelled by knot component.

        The output intentionally does not use canonical invariant factors:
        preserving component labels is more informative than merging coprime
        factors that originated in different knot summands.
        """
        if not self.invariant_factors:
            return "0"
        
        parts = []
        for comp in self._decomposition:
            # Flatten only this component so every displayed factor can retain
            # the label of the knot summand that produced it.
            factors = self._get_component_factors(comp)
            
            if not factors:
                continue
                
            group_str = " \u2295 ".join([f"Z/{f}Z" if f != 0 else "Z" for f in factors])
            
            # Rebuild the iterated-torus-knot label and restore the signed-sum
            # marker.  Sign affects the label but not the group factors.
            desc_str = "T(" + "; ".join([f"{p},{q}" for p, q in comp['description']]) + ")"
            sign_str = "-" if comp['sign'] == -1 else ""
            
            parts.append(f"({group_str})[{sign_str}{desc_str}]")
            
        return " \u2295 ".join(parts)
        
    def __repr__(self):
        """Return a concise representation identifying the knot and degree."""
        return f"BranchedCoverHomology(knot='{self.knot}', N={self.cover_degree})"


class BranchedCoverHomologyElement:
    """An element of a :class:`BranchedCoverHomology` group.

    Internally, an element is a flat integer coordinate list paired with
    ``homology.all_invariant_factors``.  A coordinate paired with ``m > 1`` is
    stored modulo ``m`` in the standard range ``0, ..., m-1``.  A coordinate
    paired with factor zero belongs to a free ``Z`` summand and is therefore
    stored without modular reduction.

    Callers may provide coordinates in either of two forms:

    * a flat list in structural factor order; or
    * ``[component][layer][coordinate]`` nesting that mirrors the homology
      decomposition.  A layer's coordinate list already includes all of its
      repeated copies, so multiplicity is not represented by another nesting
      level.

    Normalizing to one flat representation makes arithmetic independent of
    the input spelling while retaining the component-and-layer interpretation
    through the parent homology object.
    """

    def __init__(self, homology, values):
        """Validate, flatten, and normalize a coordinate description.

        Args:
            homology: The parent :class:`BranchedCoverHomology` group.
            values: One integer for a one-generator group, a flat coordinate
                list, or values nested by connected-sum component and layer.

        Raises:
            TypeError: If the parent or outer value container has the wrong
                type.
            ValueError: If the number or nesting of coordinates does not match
                the parent group's structural decomposition.
        """
        if type(homology).__name__ != 'BranchedCoverHomology':
            raise TypeError(f"Expected a BranchedCoverHomology object, got {type(homology)}.")
            
        self._homology = homology
        # This factor order is the contract used for both flattening and every
        # subsequent coordinatewise group operation.
        factors = homology.all_invariant_factors
        
        # A scalar is convenient and unambiguous only for a one-generator
        # group.  It must not be silently broadcast over several generators.
        if isinstance(values, (int, Integer)):
             if len(factors) == 1:
                 flat_values = [values]
             else:
                 raise ValueError(f"Single value provided but homology has {len(factors)} generators.")
        elif isinstance(values, (list, tuple)):
            # Accept an already-flat coordinate list without losing tuple
            # compatibility.  Lists or tuples inside it signal nested input.
            if len(values) == len(factors) and all(not isinstance(v, (list, tuple)) for v in values):
                flat_values = values
            else:
                # Nested input follows components -> layers -> coordinates.
                flat_values = self._flatten_nested_values(values, homology)
        else:
            raise TypeError(f"Values must be an integer or a list of integers. Got {type(values)}.")

        # Keep this final check even after structural flattening: it protects
        # the flat path and provides a single invariant before normalization.
        if len(flat_values) != len(factors):
            raise ValueError(f"Value mismatch: Homology has {len(factors)} generators, but {len(flat_values)} values were provided.")
            
        # Finite coordinates use canonical nonnegative residues.  A zero factor
        # denotes Z, so its coordinate is converted to ``Integer`` but left
        # unreduced.  Reconstructing results through this initializer means all
        # arithmetic operations inherit exactly the same normalization rule.
        self._values = [Integer(v) % f if f != 0 else Integer(v) for v, f in zip(flat_values, factors)]

    @staticmethod
    def _flatten_nested_values(nested_values, homology):
        """Flatten ``[component][layer][coordinate]`` structural input.

        Empty coordinate lists are meaningful for layers with no nontrivial
        Smith factors.  They must still occupy their layer position so that
        the nesting remains aligned with the knot decomposition.
        """
        # ``decomposition`` is a defensive snapshot.  Only its shape and layer
        # metadata are inspected; the homology object is never mutated.
        if len(nested_values) != len(homology.decomposition):
            raise ValueError(f"Summand count mismatch: Expected {len(homology.decomposition)}, got {len(nested_values)}.")
            
        flat = []
        for c_idx, (comp_data, comp_values) in enumerate(zip(homology.decomposition, nested_values)):
            layers = comp_data['layers']
            # Even a layer contributing no generators requires an explicit
            # empty list, making missing layers distinguishable from zero data.
            if len(comp_values) != len(layers):
                raise ValueError(f"Layer count mismatch in Summand {c_idx}: Expected {len(layers)}, got {len(comp_values)}.")
                
            for l_idx, (layer, layer_values) in enumerate(zip(layers, comp_values)):
                multiplicity = layer['multiplicity']
                base_factors = layer['base_factors']
                # Repeated copies are flattened consecutively within the layer,
                # so the caller supplies one coordinate per repeated factor.
                expected_count = multiplicity * len(base_factors)
                
                if len(layer_values) != expected_count:
                    raise ValueError(f"Value mismatch in Summand {c_idx}, Layer {l_idx}: Expected {expected_count} values, got {len(layer_values)}.")
                
                flat.extend(layer_values)
        return flat


    # ------------------------------------------------------------------
    # Parent and coordinate access
    # ------------------------------------------------------------------

    @property
    def homology(self):
        """Return the parent homology group for these coordinates."""
        return self._homology

    @property
    def values(self):
        """Return a copy of the normalized, flattened coordinate list."""
        # The copy prevents a caller from bypassing modular normalization by
        # mutating the returned list in place.
        return list(self._values)

    @property
    def is_torsion(self):
        """Return whether this element has finite order.

        In the invariant-factor convention used here, a positive factor
        denotes a finite cyclic summand and factor zero denotes a free Z
        summand.  Coordinates in finite summands are automatically torsion;
        therefore only the coordinates paired with zero factors need testing.
        """
        factors = self._homology.all_invariant_factors
        for v, f in zip(self._values, factors):
            # Any nonzero free coordinate gives the whole element infinite
            # order, regardless of its coordinates in finite cyclic summands.
            if f == 0 and v != 0:
                return False

        # All free coordinates vanish, so the element lies in the direct sum
        # of the finite cyclic factors (including the zero element).
        return True


    # ------------------------------------------------------------------
    # Group arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        """Add coordinates of two elements belonging to the same parent."""
        if not isinstance(other, BranchedCoverHomologyElement):
            raise TypeError("Can only add another BranchedCoverHomologyElement.")
        if self._homology != other._homology:
            raise ValueError("Cannot add elements from different homology groups.")
            
        # The constructor reduces finite coordinates and leaves free ones as
        # ordinary integer sums.
        new_values = [v1 + v2 for v1, v2 in zip(self._values, other._values)]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __sub__(self, other):
        """Subtract coordinates of two elements belonging to the same parent."""
        if not isinstance(other, BranchedCoverHomologyElement):
            raise TypeError("Can only subtract another BranchedCoverHomologyElement.")
        if self._homology != other._homology:
            raise ValueError("Cannot subtract elements from different homology groups.")
            
        # Reconstructing through ``__init__`` reduces negative differences in
        # finite summands while retaining them unchanged in free summands.
        new_values = [v1 - v2 for v1, v2 in zip(self._values, other._values)]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __neg__(self):
        """Return the additive inverse, normalized in every finite summand."""
        # The constructor turns -v into its canonical residue modulo each
        # positive factor and leaves ordinary integer negatives on free factors.
        new_values = [-v for v in self._values]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __mul__(self, scalar):
        """Multiply every coordinate by a Python or Sage integer."""
        if not isinstance(scalar, (int, Integer)):
            raise TypeError("Scalar multiplication only supported for integers.")
        # Normalization is intentionally centralized in the constructor rather
        # than duplicated for left and right scalar multiplication.
        new_values = [v * scalar for v in self._values]
        return BranchedCoverHomologyElement(self._homology, new_values)

    def __rmul__(self, scalar):
        """Support integer scalar multiplication written on the left."""
        return self.__mul__(scalar)

    def __eq__(self, other):
        """Compare parent identity and normalized structural coordinates.

        ``BranchedCoverHomology`` currently has identity-based equality, so
        elements constructed from two separately computed but equal-looking
        groups do not compare equal.  This prevents coordinates from being
        identified without an explicit isomorphism between their parents.
        """
        if not isinstance(other, BranchedCoverHomologyElement):
            return False
        return self._homology == other._homology and self._values == other._values

    def __repr__(self):
        """Return an unambiguous representation including the parent group."""
        return f"BranchedCoverHomologyElement(homology={repr(self._homology)}, values={self._values})"

    def __str__(self):
        """Display the normalized coordinates in structural factor order."""
        return str(self._values)
