"""Core data model for generalized algebraic knots.

The package represents a generalized algebraic knot by a signed connected sum
of iterated positive torus knots.  Its concrete description has the form::

    [
        (sign_0, [(p_00,q_00), (p_01,q_01), ...]),
        (sign_1, [(p_10,q_10), (p_11,q_11), ...]),
        ...,
    ]

The outer list is ordered connected-sum data.  A sign ``+1`` denotes the
listed knot and ``-1`` denotes its concordance inverse.  Within one summand,
the first pair describes the innermost/base torus knot; every later pair is a
new cabling pattern, so the final pair is the outermost cabling operation.
Each ``p`` and ``q`` is an integer greater than one and the two are coprime.

This is deliberately a structural representation rather than a canonical
normal form in the knot concordance group.  Connected sum concatenates
descriptions, and negation flips signs.  In particular, ``K # -K`` retains two
visible summands rather than being simplified away.  Preserving that data is
important to downstream invariants that decompose calculations by summand and
satellite layer.

The object owns a normalized nested-list copy of its input and returns fresh
copies through ``description``.  Callers may therefore use lists or tuples at
construction time without subsequently being able to mutate the stored knot.
"""

from sage.all import Integer, gcd


class GeneralizedAlgebraicKnot:
    r"""A signed connected sum of iterated positive torus knots.

    The knot is described by a nested list/tuple structure::

        [(sign_1, knot_desc_1), (sign_2, knot_desc_2), ...]

    At the top level, each pair is one connected-sum component.  ``sign`` is
    ``+1`` for the listed knot and ``-1`` for its orientation-reversed mirror,
    representing the concordance inverse.  ``knot_desc`` is a nonempty cable
    sequence::

        [(p_1,q_1), (p_2,q_2), ..., (p_n,q_n)]

    The sequence runs from the innermost/base torus knot to the outermost
    cabling pattern.  Each pair represents a positive torus-knot pattern
    ``T(p,q)`` and satisfies ``p,q > 1`` and ``gcd(p,q) = 1``.

    The stored signs affect additive invariants such as the
    Levine--Tristram signature.  They do not affect the normalized Alexander
    polynomial used by this package, since a knot and its concordance inverse
    have the same Alexander polynomial up to the conventional units.
    """

    def __init__(self, desc):
        """Validate and copy a generalized algebraic knot description.

        Args:
            desc: A nonempty list or tuple of ``(sign, cable_sequence)`` pairs.
                Cable sequences and cable pairs may also be lists or tuples.

        Raises:
            TypeError: If a required container or cable parameter has the
                wrong type.
            ValueError: If the structure, sign, positivity, or coprimality
                requirements are violated.
        """
        # Validate the complete hierarchy before publishing a partially
        # initialized object.
        self.__class__.verify_description(desc)
        # Normalize every accepted container spelling to owned lists.  Cable
        # pairs themselves are immutable tuples, while both surrounding list
        # levels are recreated to prevent aliasing the caller's input.
        self._desc = [
            (sign, [(p, q) for p, q in knot_desc])
            for sign, knot_desc in desc
        ]


    # ------------------------------------------------------------------
    # Description validation and construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def verify_description(desc):
        """Validate every level of a generalized knot description.

        Validation preserves precise component and cable indices in error
        messages, which is particularly useful for long iterated descriptions.
        Python ``int`` and Sage ``Integer`` parameters are both supported.

        Returns:
            ``True`` after the complete description has been validated.

        Raises:
            TypeError: If the outer description, a cable sequence, or a cable
                parameter has the wrong type.
            ValueError: If the description is empty, a structural pair has the
                wrong length, a sign is not ``+1`` or ``-1``, or cable
                parameters fail positivity or coprimality.
        """
        # Lists and tuples are accepted throughout the public representation;
        # other iterables are not consumed implicitly.
        if not isinstance(desc, (list, tuple)):
            raise TypeError(f"The description must be a list or tuple. Got {type(desc)}.")

        if len(desc) == 0:
            raise ValueError("The knot description must contain at least one summand.")

        for i, element in enumerate(desc):
            # Each top-level entry must separate the concordance sign from one
            # iterated-torus-knot description.
            if not isinstance(element, (list, tuple)) or len(element) != 2:
                raise ValueError(f"Element at index {i} must be a pair (sign, knot_description).")
            
            sign, knot_desc = element

            # The model has no empty-knot summand: even an ordinary torus knot
            # requires one cable pair.
            if len(knot_desc) == 0:
                raise ValueError(
                    f"Knot description at index {i} must contain at least one cabling pair."
                )

            # Signs encode a summand or its concordance inverse; arbitrary
            # integer coefficients are not part of this representation.
            if sign not in (1, -1):
                raise ValueError(f"Sign at index {i} must be 1 or -1. Got {sign}.")

            # Retain the same explicit list/tuple contract at the cable level.
            if not isinstance(knot_desc, (list, tuple)):
                raise TypeError(f"Knot description at index {i} must be a list or tuple.")

            # Walk from the base torus knot to successive outer cable patterns.
            for j, cable_pair in enumerate(knot_desc):
                if not isinstance(cable_pair, (list, tuple)) or len(cable_pair) != 2:
                    raise ValueError(f"Cable parameter at index {i}, sub-index {j} must be a pair (p, q).")
                
                p, q = cable_pair
                
                # Sage calculations frequently supply ``Integer`` objects, so
                # accept them alongside ordinary Python integers.
                if not isinstance(p, (int, Integer)) or not isinstance(q, (int, Integer)):
                    raise TypeError(f"Parameters p and q must be integers. Got {type(p)}, {type(q)} at index {i}, {j}.")
                    
                # Positive torus-knot patterns in this model require both
                # parameters to be strictly greater than one.
                if p <= 1 or q <= 1:
                    raise ValueError(f"Parameters p and q must be > 1. Got ({p}, {q}) at index {i}, {j}.")

                # Coprimality ensures T(p,q) is a knot rather than a multi-
                # component torus link.
                if gcd(p, q) != 1:
                    raise ValueError(f"Parameters p and q must be relatively prime. Got gcd({p}, {q}) != 1 at index {i}, {j}.")
        
        return True


    @property
    def description(self):
        """Return a fresh nested copy of the normalized description.

        Mutating either the input used at construction time or this returned
        list cannot change the knot stored by the object.
        """
        return [
            (sign, list(knot_desc)) for sign, knot_desc in self._desc
        ]

    @classmethod
    def torus_knot(cls, p, q, sign=1):
        """Construct the one-summand torus knot ``sign*T(p,q)``.

        Parameter and sign validation is delegated to the main constructor so
        this convenience path has exactly the same contract as a handwritten
        description.
        """
        return cls([(sign, [(p, q)])])

    @classmethod
    def iterated_torus_knot(cls, cable_sequence, sign=1):
        """Construct one signed iterated torus knot from a cable sequence.

        ``cable_sequence`` is ordered from the innermost/base torus knot to the
        outermost cabling pattern.  It is copied before being incorporated into
        the object's normalized description.
        """
        return cls([(sign, list(cable_sequence))])

    def cable(self, p, q):
        """Return the ``(p,q)``-cable of a one-summand knot.

        The new pair is appended to the cable sequence and therefore becomes
        its outermost pattern.  The original component sign is preserved.
        Cabling a connected sum is intentionally unsupported because this data
        model would not specify which summand should be cabled.
        """
        if len(self) != 1:
            raise ValueError("Cabling is only supported for iterated torus knots (single summand).")

        sign, knot_desc = self._desc[0]
        # Re-enter through the constructor so the new parameters receive the
        # same integer, positivity, and coprimality checks as initial input.
        return type(self)([(sign, list(knot_desc) + [(p, q)])])


    # ------------------------------------------------------------------
    # Knot invariants
    # ------------------------------------------------------------------

    def signature(self):
        """Return the Levine--Tristram signature function of the knot.

        The invariant implementation applies the satellite formula within each
        iterated summand, changes sign for concordance inverses, and adds the
        functions of connected-sum components.
        """
        # Import locally to keep the core description model independent of the
        # heavier signature machinery until the invariant is requested.
        from gaknot.invariants.LT_signature import LT_signature_generalized_algebraic_knot
        return LT_signature_generalized_algebraic_knot(self.description)

    def alexander_polynomial(self):
        """Return the normalized Alexander polynomial over ``ZZ[t]``.

        For a cable ``K_(p,q)``, the implementation uses

        ``Delta_{K_(p,q)}(t) = Delta_{T(p,q)}(t) * Delta_K(t^p)``.

        Iterating this identity follows each cable sequence from inside out.
        Connected sum becomes multiplication.  The component sign is
        intentionally ignored: a knot and its concordance inverse have the
        same Alexander polynomial up to units, and the utility function uses a
        fixed normalization.
        """
        from sage.all import PolynomialRing, ZZ
        from gaknot.utils.utility import alexander_polynomial_iterated_knot
        
        # Coerce the running product into the same integer polynomial ring as
        # each summand invariant, beginning with the multiplicative identity.
        R = PolynomialRing(ZZ, 't')
        total_poly = R(1)
        
        for sign, knot_desc in self.description:
            # ``sign`` matters to concordance-additive invariants but not to
            # this normalized Alexander polynomial.
            poly = alexander_polynomial_iterated_knot(knot_desc)
            total_poly *= poly
            
        return total_poly


    # ------------------------------------------------------------------
    # Shape predicates used to select specialized invariant formulas
    # ------------------------------------------------------------------

    def is_positive_torus_knot(self):
        """Return whether this is one positive, one-pair torus-knot summand."""
        # A connected sum is never classified as a single torus knot, even if
        # every one of its components is individually a positive torus knot.
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == 1 and len(knot_desc) == 1

    def is_negative_torus_knot(self):
        """Return whether this is the inverse of one ordinary torus knot."""
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == -1 and len(knot_desc) == 1

    def is_iterated_torus_knot(self):
        """Return whether this is one positive iterated torus knot.

        An ordinary positive torus knot is the one-layer special case and
        therefore also satisfies this broader predicate.
        """
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == 1 and len(knot_desc) >= 1

    def is_neg_iterated_torus_knot(self):
        """Return whether this is one negative iterated torus knot.

        As above, the predicate includes an ordinary one-layer negative torus
        knot but excludes every connected sum.
        """
        if len(self) != 1:
            return False
            
        sign, knot_desc = self._desc[0]
        return sign == -1 and len(knot_desc) >= 1


    # ------------------------------------------------------------------
    # Connected-sum and concordance operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        """Return the connected sum by concatenating component descriptions.

        No cancellation or canonical reordering is attempted: structural
        component order is observable to downstream decomposition code.
        """
        if not isinstance(other, GeneralizedAlgebraicKnot):
            raise TypeError("Can only add another GeneralizedAlgebraicKnot.")

        return type(self)(self.description + other.description)

    def __neg__(self):
        """Return the concordance inverse by flipping every component sign.

        Component order and all cable parameters remain unchanged.
        """
        new_knot_desc = [(-sign, knot_desc) for sign, knot_desc in self.description]
        return GeneralizedAlgebraicKnot(new_knot_desc)

    def __sub__(self, other):
        """Return the connected sum of ``self`` with the inverse of ``other``."""
        return self + (-other)


    # ------------------------------------------------------------------
    # Connected-sum container interface
    # ------------------------------------------------------------------

    def __len__(self):
        """Return the number of connected-sum components, not cable layers."""
        return len(self._desc)
    

    def __getitem__(self, i):
        """Return selected connected-sum components as a new knot object.

        Integer indexing supports the usual negative Python indices and wraps
        the selected component in a one-summand knot.  Slicing returns a knot
        with the selected summands and supports Python's usual step semantics.
        An empty slice is rejected by the constructor because this model has
        no empty-knot description.
        """
        try:
            # Integer access preserves the outer component pair and constructs
            # a new independent one-summand object around it.
            if isinstance(i, (int, Integer)):
                return GeneralizedAlgebraicKnot([self._desc[int(i)]])
            
            # Slicing delegates Python's start/stop/step semantics to the
            # stored list, then validates the resulting structural description.
            elif isinstance(i, slice):
                return GeneralizedAlgebraicKnot(self._desc[i])
                
            else:
                raise TypeError(f"Invalid argument type: {type(i)}")
                
        except IndexError:
            # Replace the raw list error with domain context about summands.
            raise IndexError(f"Knot index out of range. The knot has {len(self)} summand(s).")
    

    # ------------------------------------------------------------------
    # Text representations
    # ------------------------------------------------------------------

    @staticmethod
    def _it_torus_knot_desc_to_txt(desc):
        """Format one inner-to-outer cable sequence as ``T(...; ...)``.

        Example: ``[(2,3), (6,5)]`` becomes ``'T(2,3; 6,5)'``.
        """
        return 'T(' + '; '.join([f"{p},{q}" for p, q in desc]) + ')'

    def __str__(self):
        """Return a human-readable signed connected-sum description.

        Semicolons separate successive cabling pairs and ``#`` denotes
        connected sum.  For example::

            T(2,3; 2,5; 3,4) # -T(5,2; 3,7)

        The output reflects stored structure verbatim; inverse pairs are not
        cancelled and connected-sum components are not reordered.
        """
        components = []
        for sign, knot_desc in self.description:
            # Positive summands need no prefix; negative signs mark the
            # concordance inverse of the complete iterated summand.
            prefix = "-" if sign == -1 else ""
            knot_str = self._it_torus_knot_desc_to_txt(knot_desc)
            components.append(f"{prefix}{knot_str}")
        
        return ' # '.join(components)

    def __repr__(self):
        """Return the complete normalized data needed to recreate the object.

        Unlike ``str``, this representation exposes the enclosing class and
        the explicit signed nested description.
        """
        return f"GeneralizedAlgebraicKnot({self.description})"

    
