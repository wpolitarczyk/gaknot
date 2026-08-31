"""Characters on the torsion subgroup of branched-cover homology.

For a generalized algebraic knot ``K`` and cover degree ``N``, this module
represents homomorphisms

    chi: Tor(H_1(Sigma_N(K); Z)) -> Q/Z.

The parent :class:`BranchedCoverHomology` supplies a structural decomposition
into connected-sum components, outer-to-inner satellite layers, repeated
copies of each layer, and cyclic generators.  ``Character`` accepts values in
that hierarchy and then stores them as one flat list in the same order as
``homology.all_invariant_factors``.

If a generator has modulus ``m > 1``, its image ``x`` is well-defined exactly
when ``m*x`` is integral, because integers represent zero in ``Q/Z``.  A zero
modulus denotes a free ``Z`` summand.  This class models only a character on
the torsion subgroup, so it requires the coordinate corresponding to every
free summand to be zero.  All accepted values are converted to exact Sage
rationals and normalized to their unique representatives in ``[0,1)``.

Constructor input has shape ``[component][layer][coordinate]``.  Coordinates
from repeated copies of one layer are concatenated in copy order.  The
``restrict_to_layer`` method reverses that last flattening step and returns
``[copy][coordinate]`` for the requested structural layer.
"""

from sage.all import QQ, ZZ, Rational


class Character:
    """A torsion-supported character with values in ``Q/Z``.

    The class stores the character as the images of the invariant-factor
    generators in structural order.  A positive invariant factor ``m``
    describes ``Z/mZ``, so the corresponding image ``x`` must satisfy
    ``m*x = 0`` in ``Q/Z``.  A zero factor describes a free ``Z`` summand and
    must receive the value zero.
    
    Input values follow the structural hierarchy of the knot::

        [
          [                       # connected-sum component 0
            [value, ...],         # layer 0 (outermost)
            [value, ...],         # layer 1
          ],
          [                       # connected-sum component 1
            ...
          ],
        ]

    A layer with multiplicity ``r`` and ``s`` base factors requires ``r*s``
    entries in its single value list.  Entries for the first copy come first,
    followed by entries for the second copy, and so on.
    """

    def __init__(self, homology, nested_values):
        """Validate and flatten a structurally nested character description.

        Args:
            homology: The :class:`BranchedCoverHomology` on which the character
                is defined.
            nested_values: Rational values nested as components, layers, then
                a flat coordinate list containing all copies of that layer.

        Raises:
            TypeError: If the parent has the wrong type or a coordinate cannot
                be interpreted as a rational number.
            ValueError: If the nested shape disagrees with the homology
                decomposition, a finite value is incompatible with its cyclic
                modulus, or a free coordinate is nonzero.
        """
        # A character derives all coordinate counts and moduli from its parent;
        # accepting an unrelated object would make later flattening ambiguous.
        if type(homology).__name__ != 'BranchedCoverHomology':
            raise TypeError(f"Expected a BranchedCoverHomology object, got {type(homology)}.")
            
        self._homology = homology
        # Validated coordinates are appended in precisely the structural order
        # used by ``homology.all_invariant_factors``.
        self._values = []

        # The outer list must retain even components whose homology happens to
        # be trivial; structural positions cannot be inferred from coordinates.
        if len(nested_values) != len(homology.decomposition):
            raise ValueError(
                f"Input structure mismatch: Knot has {len(homology.decomposition)} connected sum components, "
                f"but input list has {len(nested_values)} items."
            )

        # Descend in the same order used by the homology decomposition:
        # connected-sum components first, then outer-to-inner layers.
        for c_idx, (comp_data, comp_values) in enumerate(zip(homology.decomposition, nested_values)):
            
            # A layer remains structurally present even if ``base_factors`` is
            # empty, so the caller must supply an explicit empty value list.
            layers = comp_data['layers']
            if len(comp_values) != len(layers):
                raise ValueError(
                    f"Structure mismatch in Component {c_idx}: Expected {len(layers)} layers, "
                    f"but got {len(comp_values)} value lists."
                )

            for l_idx, (layer, layer_values) in enumerate(zip(layers, comp_values)):
                
                # One copy has one coordinate per base factor.  Satellite
                # recursion may produce several identical copies, all written
                # consecutively inside this layer's single input list.
                multiplicity = layer['multiplicity']
                base_factors = layer['base_factors']
                expected_count = multiplicity * len(base_factors)
                
                if len(layer_values) != expected_count:
                    raise ValueError(
                        f"Value mismatch in Component {c_idx}, Layer {l_idx}: "
                        f"Expected {expected_count} values ({multiplicity} copies x {len(base_factors)} factors), "
                        f"but received {len(layer_values)}."
                    )
                
                # Walk the flat layer input in copy-major order.  For two
                # copies with factors [m_1,m_2], the expected sequence is
                # [copy1_gen1, copy1_gen2, copy2_gen1, copy2_gen2].
                val_ptr = 0
                for _ in range(multiplicity):
                    for modulus in base_factors:
                        raw_val = layer_values[val_ptr]
                        
                        # Work in QQ so compatibility and reduction modulo one
                        # are exact rather than floating-point computations.
                        try:
                            rational_val = QQ(raw_val)
                        except (TypeError, ValueError):
                            raise TypeError(
                                f"Invalid character value in Comp {c_idx}, Layer {l_idx}. "
                                f"Value must be rational. Got {raw_val}."
                            )

                        # The relation m*g = 0 in Z/mZ must remain true after
                        # applying chi.  Thus m*chi(g) must lie in Z, which is
                        # the zero class in Q/Z.
                        if modulus != 0:
                            if not (rational_val * modulus).is_integer():
                                raise ValueError(
                                    f"Invalid value in Comp {c_idx}, Layer {l_idx}. "
                                    f"Value {rational_val} is not compatible with Z/{modulus}Z."
                                )
                        else:
                            # A zero Smith factor denotes a free generator.  A
                            # general homomorphism Z -> Q/Z could be nonzero,
                            # but this class represents only the restriction to
                            # Tor(H_1), so the stored extension is forced to zero.
                            if rational_val != 0:
                                raise ValueError(
                                    f"Invalid value in Comp {c_idx}, Layer {l_idx}. "
                                    f"Characters must be zero on the torsion-free part (modulus 0). Got {rational_val}."
                                )

                        # Subtracting floor(x) does not change the class in Q/Z
                        # and chooses its unique representative in [0,1).  In
                        # particular, negative and greater-than-one inputs are
                        # normalized before becoming observable through values.
                        normalized_val = rational_val - rational_val.floor()
                        self._values.append(normalized_val)
                        val_ptr += 1


    # ------------------------------------------------------------------
    # Structural restriction
    # ------------------------------------------------------------------

    def restrict_to_layer(self, component_index, layer_index):
        """Return character values on one satellite layer, grouped by copy.

        Args:
            component_index: Index of the connected-sum component.
            layer_index: Outer-to-inner layer index within that component.

        Returns:
            A list ``[copy][coordinate]``.  Its outer length is the layer
            multiplicity and each inner length is the number of base factors.
            Consequently a copied layer with no generators returns one empty
            list for every copy rather than disappearing.

        Raises:
            IndexError: If either structural index is outside its valid range.
        """
        # Validate the component before looking up its layer count so each
        # invalid hierarchy level receives a precise error message.
        if component_index < 0 or component_index >= len(self._homology.decomposition):
            raise IndexError(f"Component index {component_index} out of range.")
            
        target_component = self._homology.decomposition[component_index]
        if layer_index < 0 or layer_index >= len(target_component['layers']):
            raise IndexError(f"Layer index {layer_index} out of range for component {component_index}.")

        # Locate the component within the global flat coordinate list by
        # skipping every generator belonging to earlier components.
        current_idx = 0
        for c_idx in range(component_index):
            comp = self._homology.decomposition[c_idx]
            current_idx += self._count_factors_in_component(comp)
            
        # Then skip earlier outer-to-inner layers of the selected component.
        # Each such layer occupies multiplicity * factors-per-copy entries.
        for l_idx in range(layer_index):
            layer = target_component['layers'][l_idx]
            current_idx += layer['multiplicity'] * len(layer['base_factors'])
            
        # Split the target layer into fresh one-copy slices.  Slicing also
        # ensures callers cannot mutate the Character's internal flat list.
        target_layer = target_component['layers'][layer_index]
        num_copies = target_layer['multiplicity']
        factors_per_copy = len(target_layer['base_factors'])
        
        layer_values = []
        for _ in range(num_copies):
            copy_values = self._values[current_idx : current_idx + factors_per_copy]
            layer_values.append(copy_values)
            current_idx += factors_per_copy
            
        return layer_values

    def induced_companion_characters(self, component_index=0):
        r"""Return the ``h`` characters in Theorem 4.19's nondivisible branch.

        For the selected component, let ``n`` be this character's cover
        degree, let ``w`` be its outer winding, and put ``h=gcd(n,w)``.  This
        method restricts the character to the ``h`` structurally recorded
        copies of ``H_1(Sigma_(n/h)(K))`` belonging to the inner companion.

        The returned diagnostic record is iterable and retains the exact
        cover, winding, companion, and deck-copy ordering.  The implementation
        is imported lazily to keep ordinary character evaluation independent
        of the satellite-transport machinery.
        """

        from gaknot.invariants.character_transport import (
            induced_companion_characters,
        )
        return induced_companion_characters(self, component_index)

    def outer_torus_pattern_phase_orbit(
        self,
        component_index=0,
        orbit_length=None,
    ):
        r"""Evaluate this character on an outer cable's distinguished orbit.

        For a standard ``(p,q)`` cable, the returned entries are the exact
        values on ``t_Q^j q_Q(mu_Q^(-p) eta)``.  They supply the root-of-unity
        phases in both nonzero-winding branches of BCP-II, Theorem 4.19.

        By default the complete orbit in the current branched cover is
        returned.  A caller implementing the nondivisible branch usually asks
        for ``gcd(cover_degree,p)`` entries; induced-character transport does
        this automatically.
        """

        from gaknot.invariants.character_transport import (
            outer_torus_pattern_phase_orbit,
        )
        return outer_torus_pattern_phase_orbit(
            self,
            component_index,
            orbit_length,
        )


    # ------------------------------------------------------------------
    # Derived invariants and small structural helpers
    # ------------------------------------------------------------------

    def twisted_alexander_polynomial(self):
        """Compute the metabelian twisted Alexander polynomial for ``self``.

        The formula is currently implemented only for positive, one-layer
        torus knots.  This convenience method obtains the knot from the parent
        homology and delegates the calculation; the target function performs
        the remaining checks, including the required relation between the
        cover degree and the torus-knot winding parameter.

        Raises:
            NotImplementedError: If the branch knot is not a positive torus
                knot supported by the current formula.
        """
        if self._homology.knot.is_positive_torus_knot():
            # Keep the specialized invariant dependency local: ordinary
            # character construction and evaluation do not need to import it.
            from gaknot.invariants.twisted_alexander import twisted_alexander_torus_knot
            return twisted_alexander_torus_knot(self._homology.knot, self)
        else:
            raise NotImplementedError("Twisted Alexander polynomial is currently only implemented for positive torus knots.")

    def _count_factors_in_component(self, component):
        """Count flat coordinates occupied by one structural component."""
        count = 0
        for layer in component['layers']:
            # Every copy contributes the complete base-factor list.
            count += layer['multiplicity'] * len(layer['base_factors'])
        return count

    @property
    def homology(self):
        """Return the parent :class:`BranchedCoverHomology` object."""
        return self._homology

    @property
    def values(self):
        """Returns a copy of the flattened list of normalized values.

        Character values are validated when the character is constructed.  A
        defensive copy prevents callers from mutating the internal list later
        and thereby bypassing the rationality, normalization, and modulus
        checks above.  Coordinates are ordered exactly like
        ``homology.all_invariant_factors``.
        """
        return list(self._values)


    # ------------------------------------------------------------------
    # Evaluation on homology elements
    # ------------------------------------------------------------------

    def __call__(self, element):
        """Evaluate the character on a torsion homology element.

        If the stored character values are ``chi_i`` and the element has
        structural coordinates ``g_i``, evaluation is

        ``sum_i chi_i*g_i (mod Z)``.

        The result is returned as the unique rational representative in
        ``[0,1)``.  Evaluation on a non-torsion element is deliberately
        undefined because this object models only a map on ``Tor(H_1)``.

        Raises:
            TypeError: If ``element`` is not a homology element.
            ValueError: If it belongs to a different parent group or has
                infinite order.
        """
        # Coordinate values are meaningful only for homology elements, whose
        # constructor also guarantees the expected number of coordinates.
        if type(element).__name__ != 'BranchedCoverHomologyElement':
            raise TypeError(f"Expected a BranchedCoverHomologyElement, got {type(element)}.")
            
        # Equal-looking coordinate lists from different covers must not be
        # paired with this character's invariant-factor decomposition.  The
        # parent also fixes the semantic meaning of every structural position.
        if element.homology != self._homology:
            raise ValueError("Character and element must belong to the same homology group.")
            
        # A nonzero free coordinate gives the element infinite order, placing
        # it outside the domain represented by this Character object.
        if not element.is_torsion:
            raise ValueError("Character evaluation is only defined for torsion elements.")

        # Coordinate orders agree by construction, so evaluation is the exact
        # dot product in QQ.  The final subtraction reduces its class modulo Z.
        res = 0
        for chi_i, g_i in zip(self._values, element.values):
            res += chi_i * g_i
            
        return res - res.floor()

    def __repr__(self):
        """Return a concise representation of the normalized flat values."""
        return f"Character(values={self._values})"
