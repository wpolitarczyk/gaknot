#!/usr/bin/env python3

r"""calculations of signature function and sigma invariant of generalized algebraic knots (GA-knots)


The package was used to prove Lemma 3.2 from a paper
'On the slice genus of generalized algebraic knots' Maria Marchwicka and Wojciech Politarczyk).
It contains the following submodules.
    1) main.sage - with function prove_lemma
    2) signature.sage - contains SignatureFunction class;
       it encodes twisted and untwisted signature functions
       of knots and allows to perform algebraic operations on them.
    3) cable_signature.sage - contains the following classes:
        a) CableSummand - it represents a single cable knot,
        b) CableSum - it represents a cable sum, i. e. linear combination of single cable knots;
           since the signature function and sigma invariant are additive under connected sum,
           the class use calculations from CableSummand objects,
        c) CableTemplate - it represents a scheme for a cable sums.
    4) LT-signature.sage - functions for computing LT-signature functions for iterated torus knots.
"""


from .utility import import_sage
import os
import logging

package = __name__
current_file_path = os.path.abspath(__file__)
path = os.path.dirname(os.path.dirname(current_file_path))

logging.debug(f"__init__.py: current_file_path={current_file_path}")
logging.debug(f"__init__.py: calculated path={path}")

def safe_import(module, package, path):
    """Import a sage module only if the file exists, avoiding noise."""
    sage_file = os.path.join(path, package, f"{module}.sage")
    py_file = os.path.join(path, package, f"{module}.py")
    logging.debug(f"safe_import: checking for {sage_file}")
    if os.path.exists(sage_file) or os.path.exists(py_file):
        return import_sage(module, package=package, path=path)
    return None

# Ensure these match your filenames (use underscores, not hyphens)
safe_import('signature', package, path)
# safe_import('cable_signature', package, path)
# safe_import('main', package, path)
safe_import('LT_signature', package, path)
# Special case for the main gaknot module
safe_import('gaknot', package, path)
safe_import('H1_branched_cover', package, path)

# from .main import prove_lemma


# EXAMPLES::
#
# sage: eval_cable_for_null_signature([[1, 3], [2], [-1, -2], [-3]])
#
# T(2, 3; 2, 7) # T(2, 5) # -T(2, 3; 2, 5) # -T(2, 7)
# Zero cases: 1
# All cases: 1225
# Zero theta combinations:
# (0, 0, 0, 0)
#
# sage:
#
# The numbers given to the function eval_cable_for_null_signature are k-values
# for each component/cable in a direct sum.
#
# To calculate signature function for a knot and a theta value, use function
# get_signature_as_function_of_theta (see help/docstring for details).
#
# About notation:
# Cables that we work with follow a schema:
#     T(2, q_1; 2, q_2; 2, q_4) # -T(2, q_2; 2, q_4) #
#             # T(2, q_3; 2, q_4) # -T(2, q_1; 2, q_3; 2, q_4)
# In knot_formula each k[i] is related with some q_i value, where
# q_i = 2*k[i] + 1.
# So we can work in the following steps:
# 1) choose a schema/formula by changing the value of knot_formula
# 2) set each q_i all or choose range in which q_i should varry
# 3) choose vector v / theta vector.
#
