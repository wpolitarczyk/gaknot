#!/usr/bin/env sage -python

"""Measure construction and evaluation of a large signature function.

This is a benchmark rather than a test: timings depend on the processor, Sage
version, system load, and cache state, so they are reported instead of compared
with fixed pass/fail thresholds.
"""

import argparse
import statistics
import time

from gaknot.invariants.LT_signature import LT_signature_torus_knot


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=91)
    parser.add_argument("--q", type=int, default=874)
    parser.add_argument("--evaluations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _print_summary(label, samples):
    print(
        f"{label}: median={statistics.median(samples):.6f}s "
        f"range={min(samples):.6f}--{max(samples):.6f}s"
    )


def main():
    args = _parse_arguments()
    if args.evaluations <= 0 or args.repeats <= 0:
        raise ValueError("evaluations and repeats must be positive")

    evaluation_points = [i / args.evaluations for i in range(args.evaluations)]
    construction_times = []
    evaluation_times = []

    for _ in range(args.repeats):
        start = time.perf_counter()
        signature = LT_signature_torus_knot(args.p, args.q)
        construction_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        for point in evaluation_points:
            signature(point)
        evaluation_times.append(time.perf_counter() - start)

    print(
        f"T({args.p}, {args.q}); {len(signature.jumps_counter)} jumps; "
        f"{args.repeats} repeats"
    )
    _print_summary("construction", construction_times)
    _print_summary(f"{args.evaluations} evaluations", evaluation_times)


if __name__ == "__main__":
    main()
