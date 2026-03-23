#!/usr/bin/env sage -python


# TBD
# print scheme and knot nicely
# read about Factory Method, variable in docstring, sage documentation,
# print calc. to output file
# decide about printing option
# make __main__?

import os
import sys

import itertools as it
import re
import numpy as np
import importlib

# constants - which invariants should be calculate
SIGMA = 0
SIGNATURE = 1

if __name__ == '__main__':
    from utility import import_sage
    package = None
    path = ''

else:
    from .utility import import_sage
    package = __name__.split('.')[0]
    # We use the path of the parent directory because import_sage appends package
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sg = import_sage('signature', package=package, path=path)
cs = import_sage('cable_signature', package=package, path=path)

# class Config:
#     def __init__(self):
#         self.f_results = os.path.join(os.getcwd(), "results.out")

class Schema:
    r"""This class stores interesting schema of cable sums.

    Cable knots sum can be given as a scheme, e.g. a scheme from the paper:
    K(p_1 , p_2 , q_1 , q_2 , q_3 ) =
        T(2, q_1; 2, p_1) # -T(2, q_2; 2, p_1) # T(2, p_1) # -T(2, q_3; 2, p_1) +
        T(2, q_2; 2, p_2) # -T(2, p_2) # T(2, q_3; 2, p_2) # -T(2, q_1; 2, p_2).
    We can represent it as nested list:
    lemma_scheme = "[[ k[5],  k[3]], " + \
                   "[ -k[1], -k[3]], " + \
                   "[         k[3]], " + \
                   "[ -k[6], -k[3]], " + \
                   "[  k[1],  k[7]], " + \
                   "[        -k[7]], " + \
                   "[  k[6],  k[7]], " + \
                   "[ -k[5], -k[7]]]",
    where each k[i] corresponds to some q_i or p_i.
    This expression will be later evaluated with k_vector.
    See k_vector setter in class CableTemplate in cable_signature.sage module.

    Remark 1
        In the paper, we used p_i and q_i to describe torus knots and cables.
        It was convenient for writing, but in all the code and documentation
        only 'q' letter is used to encode torus knots or cables.

    Remark 2
        There are two ways to set k[i] values for a scheme:
        via q_vector or k_vector.
        Both should be lists and the relation is q[i] = 2 * k[i] + 1,
        i.e. q should be an odd prime and k should be an even number such that
        2 * k + 1 is prime.
        To fill the scheme listed above we should use a list of length 8,
        and k[0] will be omitted as it is not used in the scheme.

    Remark 3
        Except for development purposes, q_vector was computed with
        a method CableTemplate.get_q_vector and flag slice=True.
        The reason for that is that we were interested only in cases
        where a specific relation for each cabling level is preserved.
        Consider a cable T(2, q_0; 2, q_1; ...;  2, q_n).
        Then for every q_i, q_(i + 1): q_(i + 1) > q_i * 4.

    """

    # scheme that is used in the paper
    lemma_scheme_a = "[ [k[5],  k[3]], " + \
                     "[ -k[1], -k[3]], " + \
                     "[         k[3]], " + \
                     "[ -k[6], -k[3]]]"

    lemma_scheme_b = "[[ k[1],  k[7]], " + \
                     "[        -k[7]], " + \
                     "[  k[6],  k[7]], " + \
                     "[ -k[5], -k[7]]]"

    lemma_scheme = "[[ k[5],  k[3]], " + \
                   "[ -k[1], -k[3]], " + \
                   "[         k[3]], " + \
                   "[ -k[6], -k[3]], " + \
                   "[  k[1],  k[7]], " + \
                   "[        -k[7]], " + \
                   "[  k[6],  k[7]], " + \
                   "[ -k[5], -k[7]]]"

    #
    # formula_long = "[[k[0],  k[5],  k[3]], " + \
    #                "[       -k[5], -k[3]], " + \
    #                "[        k[2],  k[3]], " + \
    #                "[-k[4], -k[2], -k[3]]" + \
    #                "[ k[4],  k[1],  k[7]], " + \
    #                "[       -k[1], -k[7]], " + \
    #                "[        k[6],  k[7]], " + \
    #                "[-k[0], -k[6], -k[7]]]"
    #
    # formula_a = "[[k[0],  k[5],  k[3]], " + \
    #             "[       -k[1], -k[3]], " + \
    #             "[               k[3]], " + \
    #             "[-k[4], -k[6], -k[3]]]"
    #
    # formula_b = "[[k[4],  k[1],  k[7]], " + \
    #             "[              -k[7]], " + \
    #             "[        k[6],  k[7]], " + \
    #             "[-k[0], -k[5],  -k[7]]]"
    #
    # formula_a = "[[ k[0],  k[5],  k[3]], " + \
    #             "[        -k[1], -k[3]], " + \
    #             "[         k[2],  k[3]], " + \
    #             "[ -k[0], -k[2], -k[3]]]"
    #
    # formula_b = "[[ k[4],  k[1],  k[7]], " + \
    #             "[        -k[5], -k[7]], " + \
    #             "[         k[6],  k[7]], " + \
    #             "[ -k[4], -k[6], -k[7]]]"
    #
    # formula_a = "[[ k[0], k[5],  k[3]], " + \
    #             "[       -k[5], -k[3]], " + \
    #             "[        k[2],  k[3]], " + \
    #             "[-k[4], -k[2], -k[3]]]"
    #
    # formula_b = "[[ k[4], k[1],  k[7]], " + \
    #             "[       -k[1], -k[7]], " + \
    #             "[        k[6],  k[7]], " + \
    #             "[-k[0], -k[6], -k[7]]]"
    #
    #
    # three_layers_formula_a = "[[k[0],  k[1],  k[2]],\
    #                           [        k[3],  k[4]],\
    #                           [-k[0], -k[3], -k[4]],\
    #                           [       -k[1], -k[2]]]"
    #
    # three_layers_formula_b = "[[k[0],   k[1],  k[2]],\
    #                           [               k[3]],\
    #                           [-k[0], -k[1], -k[3]],\
    #                           [              -k[2]]]"
    #
    # short_3_layers_b = "[[         k[5],  k[3]], " + \
    #                     "[        -k[1], -k[3]], " + \
    #                     "[                k[3]], " + \
    #                     "[ -k[4], -k[6], -k[3]]]"
    #
    # short_3_layers_b = "[[k[4],   k[1],  k[7]], " + \
    #                     "[               -k[7]], " + \
    #                     "[         k[6],  k[7]], " + \
    #                     "[         -k[5], -k[7]]]"
    #
    # four_summands_scheme = "[[ k[0],  k[1],  k[3]]," + \
    #                        "[        -k[1], -k[3]]," + \
    #                        "[         k[2],  k[3]]," + \
    #                        "[ -k[0], -k[2], -k[3]]]"
    #
    # two_summands_scheme = "[ [k[0], k[1], k[4]], [-k[1], -k[3]],\
    #                        [k[2], k[3]], [-k[0], -k[2], -k[4]] ]"
    # two_small_summands_scheme = "[[k[3]], [-k[3]],\
    #                             [k[3]], [-k[3]] ]"


def main(arg=None):
    try:
        limit = int(arg[1])
    except (IndexError, TypeError):
        limit = None
    conf = Config()
    cable_loop_with_details(conf)

def prove_lemma(verbose=True, details=False):

    if verbose:
        msg = "CALCULATIONS OF THE SIGMA INVARIANT\n"
        msg += "Proof of main lemma from "
        msg += "ON THE SLICE GENUS OF GENERALIZED ALGEBRAIC KNOTS\n\n"
        print(msg)

    lemma_scheme = Schema.lemma_scheme

    cable_template = cs.CableTemplate(knot_formula=lemma_scheme,
                                      verbose=verbose)
    cable_template.fill_q_vector()
    q_v = cable_template.q_vector
    cable = cable_template.cable
    if verbose:
        msg = "Let us consider a cable knot: \nK =  "
        msg += cable.knot_description + ".\n"
        msg += "It is an example of cable knots of a scheme:\n"
        msg += str(cable_template) + "."
        print(msg)

    cable.is_function_big_for_all_metabolizers(invariant=SIGMA,
                                               verbose=verbose,
                                               details=details)


def print_sigma_for_cable(verbose=True):

    lemma_scheme_a = Schema.lemma_scheme_a
    lemma_scheme_b = Schema.lemma_scheme_b
    lemma_scheme = Schema.lemma_scheme
    scheme_four = Schema.four_summands_scheme

    cable_template = cs.CableTemplate(knot_formula=lemma_scheme)
    cable_template.fill_q_vector()
    q_v = cable_template.q_vector
    print(q_v)
    print(cable_template.cable.knot_description)
    cable_a = cs.CableTemplate(knot_formula=lemma_scheme_a,
                              verbose=verbose,
                              q_vector=q_v
                             ).cable
    cable_b = cs.CableTemplate(knot_formula=lemma_scheme_b,
                              verbose=verbose,
                              q_vector=q_v
                             ).cable
    cable = cs.CableTemplate(knot_formula=lemma_scheme_a,
                              verbose=verbose,
                              q_vector=q_v
                             ).cable

    cable.plot_sigma_for_summands()
    # cable_a.plot_sigma_for_summands()
    # cable_b.plot_sigma_for_summands()


def cable_loop_with_details(verbose=True):
    # verbose = False
    lemma_scheme_a = Schema.lemma_scheme_a
    lemma_scheme_b = Schema.lemma_scheme_b
    lemma_scheme = Schema.lemma_scheme
    cable_template = cs.CableTemplate(knot_formula=lemma_scheme)

    list_of_q_vectors = []
    # for el in [2, 3, 5, 7, 11, 13]:
    for el in [2]:
        cable_template.fill_q_vector(lowest_number=el)
        q_v = cable_template.q_vector
        print(q_v)
        print(cable_template.cable.knot_description)
        cable_a = cs.CableTemplate(knot_formula=lemma_scheme_a,
                                  verbose=verbose,
                                  q_vector=q_v
                                 ).cable
        cable_b = cs.CableTemplate(knot_formula=lemma_scheme_b,
                                  verbose=verbose,
                                  q_vector=q_v
                                 ).cable
    #     print("\n")
    #     print(cable_a.knot_description)
        is_a = cable_a.is_function_big_for_all_metabolizers(invariant=cs.SIGMA,
                                                            verbose=True,
                                                            details=True)
        is_b = cable_b.is_function_big_for_all_metabolizers(invariant=cs.SIGMA,
                                                            verbose=True,
                                                            details=True)
        if is_a and is_b:
            print("sigma is big for all metabolizers")
        else:
            print("sigma is not big for all metabolizers")
        print("\n" * 3)


def few_cable_without_calc(verbose=False):

    lemma_scheme_a = Schema.lemma_scheme_a
    lemma_scheme_b = Schema.lemma_scheme_b
    lemma_scheme = Schema.lemma_scheme

    cable_template = cs.CableTemplate(knot_formula=lemma_scheme)

    list_of_q_vectors = []
    for el in [2, 3, 5, 7, 11, 13]:
        cable_template.fill_q_vector(lowest_number=el)
        q_v = cable_template.q_vector
        print(q_v)
        print(cable_template.cable.knot_description)
        cable_a = cs.CableTemplate(knot_formula=lemma_scheme_a,
                              verbose=verbose,
                              q_vector=q_v
                             ).cable
        cable_b = cs.CableTemplate(knot_formula=lemma_scheme_b,
                                  verbose=verbose,
                                  q_vector=q_v
                                 ).cable
        is_a = cable_a.is_function_big_for_all_metabolizers(invariant=sigma)
        is_b = cable_b.is_function_big_for_all_metabolizers(invariant=sigma)
        if is_a and is_b:
            print("sigma is big for all metabolizers")
        else:
            print("sigma is not big for all metabolizers")
        print("\n" * 3)


def plot_many_untwisted_signature_functions(range_tuple=(1, 10)):
    P = Primes()
    for i in range(*range_tuple):
        q = P.unrank(i)
        a = cs.CableSummand.get_untwisted_signature_function(q=q)
        a.plot()


if __name__ == '__main__':
    if '__file__' in globals():
        # skiped in interactive mode as __file__ is not defined
        main(sys.argv)
    else:
        pass
        # main()
