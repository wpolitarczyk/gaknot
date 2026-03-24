#!/usr/bin/env sage -python

import math
import os
from collections import Counter

from .utility import mod_one

from sage.all import Integer, gcd

# Internal import logic to handle package context
if __name__ == '__main__':
    from utility import import_sage
    package = None
    path = ''
else:
    from .utility import import_sage
    package = __name__.rsplit('.', 1)[0]
    # We use the path of the parent directory because import_sage appends package
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import signature module and assign to 'sg'.
sg = import_sage('signature', package=package, path=path)


def LT_signature_torus_knot(p, q):
    """
    Computes the Levine-Tristram signature function of a torus knot T(p,q).
    Optimized version using modular arithmetic (O(pq) complexity).
    """

    if not isinstance(p, (int, Integer)) or not isinstance(q, (int, Integer)):
        raise TypeError(f'Parameters p and q have to be integers.')

    if p <= 1 or q <= 1:
        raise ValueError(f'Parameters p and q must be >1.')

    if math.gcd(p, q) != 1:
        raise ValueError(f'Parameteres p and q must be relatively prime.')

    from sage.all import inverse_mod, floor
    p_inv_q = inverse_mod(p, q)
    
    counter = Counter()
    # Jump points occur at i/(pq) where p*x and q*x are not integers.
    # For x = i/(pq), p*x = i/q and q*x = i/p.
    # Non-integers means i is not a multiple of q and i is not a multiple of p.
    for i in range(1, p * q):
        if i % p != 0 and i % q != 0:
            # i = a*p + b*q mod (pq)
            # Multiplying by p_inv_q (mod q): i * p_inv_q = a * p * p_inv_q = a (mod q)
            a_val = (i * p_inv_q) % q
            # i = a_val * p + b_val * q => b_val * q = i - a_val * p
            b_val = (i - a_val * p) // q
            
            # exponent = floor(a/q) + floor(b/p) + floor(a/q + b/p)
            # Since 0 <= a_val < q, floor(a_val/q) is always 0.
            exponent = floor(b_val / p) + floor(a_val / q + b_val / p)
            counter[Integer(i) / (p * q)] = (-1) ** exponent

    return sg.SignatureFunction(counter=counter)


def reparametrize(sig_func, p):
    """
    Helper function to compute sigma(p*theta) given sigma(theta).
    This effectively 'compresses' the signature function, repeating it p times
    scaled down by 1/p.
    """
    old_counter = sig_func.jumps_counter
    new_counter = Counter()

    for x, jump_val in old_counter.items():
        for k in range(p):
            new_x = (x + k) / p
            new_counter[new_x] += jump_val
            
    return sg.SignatureFunction(counter=new_counter)

def LT_signature_iterated_torus_knot_counter(desc):
    r"""
    Helper function that returns a Counter of jumps instead of a SignatureFunction.
    """
    if not isinstance(desc, (list, tuple)):
        raise TypeError('The variable desc should be a list or tuple.')

    from sage.all import inverse_mod, floor
    total_counter = Counter()
    current_p_prod = 1

    for i in range(len(desc) - 1, -1, -1):
        p, q = desc[i]
        p_inv_q = inverse_mod(p, q)

        for j in range(1, p * q):
            if j % p != 0 and j % q != 0:
                a_val = (j * p_inv_q) % q
                b_val = (j - a_val * p) // q
                val = (-1) ** (floor(b_val / p) + floor(a_val / q + b_val / p))

                base_jump = Integer(j) / (p * q)
                if current_p_prod == 1:
                    total_counter[base_jump] += val
                else:
                    for k in range(current_p_prod):
                        total_counter[(base_jump + k) / current_p_prod] += val

        current_p_prod *= p

    return total_counter

def LT_signature_iterated_torus_knot(desc):
    r"""
    Computes the Levine-Tristram signature function of an iterated torus knot.
    """
    counter = LT_signature_iterated_torus_knot_counter(desc)
    return sg.SignatureFunction(counter=counter)

def LT_signature_generalized_algebraic_knot(desc):
    """
    Computes the Levine-Tristram signature of a generalized algebraic knot.
    Optimized to accumulate all jumps into a single Counter before creating a SignatureFunction.
    """

    if not isinstance(desc, (list, tuple)):
        raise TypeError(f'The variable desc should be a list or tuple.')

    total_counter = Counter()

    for i, element in enumerate(desc):
        if not isinstance(element, (list, tuple)) or len(element) != 2:
            raise ValueError(f'Element at index {i} must be a pair (sign, knot_description).')

        sign, knot_desc = element

        if sign != 1 and sign != -1:
            raise ValueError(f'Sign at index {i} must be 1 or -1.')

        try:
            # Directly get the counter to avoid redundant SignatureFunction creation
            component_counter = LT_signature_iterated_torus_knot_counter(knot_desc)

            if sign == 1:
                total_counter.update(component_counter)
            else:
                total_counter.subtract(component_counter)

        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid knot description at index {i}: {e}")

    return sg.SignatureFunction(counter=total_counter)


