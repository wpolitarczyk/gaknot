#!/usr/bin/env sage -python

r"""Sparse step functions used for Levine--Tristram signatures.

A signature function is stored through its discontinuities rather than through
sampled values.  If the counter has weight ``j`` at ``x``, the value changes by
``2*j`` when ``x`` is crossed.  At the discontinuity, this module uses the
midpoint convention

``sigma(x) = 2 * sum(jumps strictly before x) + j``.

Thus the left- and right-hand values are ``sigma(x)-j`` and ``sigma(x)+j``.
Jump locations live in the half-open fundamental domain ``[0, 1)``, while
evaluation arguments are reduced modulo one to make the represented function
periodic.  Sage rational keys are preserved exactly whenever callers supply
them, allowing independently produced jumps to cancel without rounding error.

``SignatureFunction`` supplies the algebra and argument transformations.
``SignaturePloter`` turns the sparse representation into Matplotlib segments
or a standalone TikZ document.
"""

import math
from copy import copy
from collections import Counter
from bisect import bisect_left
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from gaknot.utils.utility import mod_one


# ---------------------------------------------------------------------------
# Sparse signature-function representation
# ---------------------------------------------------------------------------

class SignatureFunction:
    r"""Represent a periodic signature function by its jump weights.

    ``values`` is an iterable of ``(location, jump_weight)`` pairs. Repeated
    locations are added, and zero totals are discarded. A ready-made
    ``Counter`` may instead be supplied through ``counter``.

    Internally, the counter is private because evaluation relies on sorted-key
    and cumulative-sum caches derived during construction. The public
    ``jumps_counter`` property returns a defensive copy so external mutation
    cannot make those caches disagree with the stored jumps.
    """
    def __init__(self, values=None, counter=None, plot_title=''):
        # When no counter is supplied, aggregate possibly repeated pairs in
        # values. Counter addition matches addition of signature jumps.
        if counter is None:
            counter = Counter()
            values = values or []
            for k, v in values:
                counter[k] += v

        # Copy the supplied mapping and remove cancellations before validating
        # or caching it. A zero-total location has no mathematical effect.
        counter = Counter({k : v for k, v in counter.items() if v != 0})

        # Stored keys require canonical representatives. Evaluation arguments
        # are periodic, but allowing noncanonical keys would make sorting,
        # equality, and cancellation ambiguous.
        if any(k < 0 or k >= 1 for k in counter.keys()):
                msg = "Signature function is defined on the interval [0, 1)."
                raise ValueError(msg)

        # The private counter and the arrays below form one immutable internal
        # snapshot. Algebraic operations always construct a new object.
        self._jumps_counter = counter
        self.plot_title = plot_title

        # Pay the sorting cost once. Each later evaluation needs only a binary
        # search and one prefix-sum lookup instead of scanning every jump.
        self._sorted_keys = sorted(self._jumps_counter.keys())
        self._cumulative_sums = list(itertools.accumulate(
            [self._jumps_counter[k] for k in self._sorted_keys],
            initial=0))

    @property
    def jumps_counter(self):
        """Return a defensive copy of the sparse jump representation.

        Mutating the returned ``Counter`` cannot affect this object or make its
        cached evaluation data stale.
        """
        return copy(self._jumps_counter)

    def __rshift__(self, shift):
        """Rotate every jump location forward by ``shift`` modulo one."""
        # mod_one returns locations crossing 1 to the canonical interval.
        counter = Counter({mod_one(k + shift) : v \
                          for k, v in self._jumps_counter.items()})
        return SignatureFunction(counter=counter)

    def __lshift__(self, shift):
        """Rotate every jump location backward by ``shift`` modulo one."""
        # A left shift is a right shift by the additive inverse.
        return self.__rshift__(-shift)

    def __neg__(self):
        """Return the pointwise negative by negating every jump weight."""
        counter = Counter()
        counter.subtract(self._jumps_counter)
        return SignatureFunction(counter=counter)

    def __add__(self, other):
        """Add two signature functions coefficient by coefficient."""
        counter = copy(self._jumps_counter)
        counter.update(other._jumps_counter)
        # Titles are metadata: keep the available one, or display both when
        # both summands carry a label.
        if self.plot_title and other.plot_title:
            title = self.plot_title + " + " + other.plot_title
        else:
            title = self.plot_title or other.plot_title
        return SignatureFunction(counter=counter, plot_title=title)

    def __sub__(self, other):
        """Subtract another signature function's jump weights."""
        counter = copy(self._jumps_counter)
        counter.subtract(other._jumps_counter)
        return SignatureFunction(counter=counter)

    def __mul__(self, number):
        """Multiply every jump weight by ``number``."""
        counter = Counter({k : number * v \
                          for k, v in self._jumps_counter.items()})
        return SignatureFunction(counter=counter)

    def __rmul__(self, number):
        """Support scalar multiplication with the scalar on the left."""
        return(self.__mul__(number))

    def __eq__(self, other):
        """Compare complete sparse functions while ignoring plot metadata."""
        return self._jumps_counter == other._jumps_counter

    def __str__(self):
        """List one sorted ``location: weight`` pair per line."""
        result = ''.join([str(jump_arg) + ": " + str(jump) + "\n"
                for jump_arg, jump in sorted(self._jumps_counter.items())])
        return result

    def __repr__(self):
        """Return a compact sorted representation of all jump weights."""
        result = ''.join([str(jump_arg) + ": " + str(jump) + ", "
                for jump_arg, jump in sorted(self._jumps_counter.items())])
        # Trimming an empty result used to produce the meaningless string ".".
        if not result:
            return "SignatureFunction()"
        return result[:-2] + "."

    def __call__(self, arg):
        """Evaluate using periodicity and the midpoint convention.

        ``bisect_left`` counts jumps strictly before the reduced argument. The
        counter lookup then adds one copy of a jump at the argument itself,
        placing the value halfway between its two one-sided limits.
        """
        # Signature functions live on the unit circle, so negative and large
        # arguments are valid and are first reduced to [0, 1).
        x = mod_one(arg)
        idx = bisect_left(self._sorted_keys, x)

        # The prefix array has an initial zero. Index idx therefore gives the
        # sum of exactly the jumps whose locations are strictly less than x.
        sum_before = self._cumulative_sums[idx]

        # Counter returns zero at an ordinary point that is not a stored key.
        return 2 * sum_before + self._jumps_counter[x]

    def double_cover(self):
        r"""Return the pullback ``theta -> sigma(2*theta)``.

        A jump at ``k`` has the two preimages ``k/2`` and ``(1+k)/2`` in the
        unit interval, both carrying the original weight.
        """
        items = self._jumps_counter.items()
        counter = Counter({(1 + k) / 2 : v for k, v in items})
        counter.update(Counter({k / 2 : v for k, v in items}))
        return SignatureFunction(counter=counter)

    def square_root(self):
        """Rescale jumps in the first half of the circle by a factor of two."""
        counter = Counter()
        for jump_arg, jump in self._jumps_counter.items():
            # The strict boundary assigns a jump at 1/2 to the other branch and
            # prevents the two branch methods from duplicating it.
            if jump_arg < 1/2:
                counter[2 * jump_arg] = jump
        return SignatureFunction(counter=counter)

    def minus_square_root(self):
        """Rescale jumps in the second half of the circle by a factor of two."""
        items = self._jumps_counter.items()
        # Doubling sends [1/2, 1) to [1, 2); mod_one restores canonical keys.
        counter = Counter({mod_one(2 * k) : v for k, v in items if k >= 1/2})
        return SignatureFunction(counter=counter)

    def is_zero_everywhere(self):
        """Return whether the sparse representation has no nonzero jump."""
        # Construction removes zero coefficients, while any() also states the
        # intended mathematical predicate explicitly.
        return not any(self._jumps_counter.values())

    def extremum(self, limit=math.inf):
        """Return the first jump producing the largest absolute value.

        The reported value is the right-hand value after crossing the jump. If
        ``limit`` is finite, scanning stops when a new record exceeds it; this
        supplies a quick witness when the exact global extremum is unnecessary.
        """
        max_point = (0, 0)
        current = 0
        items = sorted(self._jumps_counter.items())
        for arg, jump in items:
            # Crossing a stored weight changes the function by twice that
            # weight under the convention used throughout this class.
            current += 2 * jump
            # Midpoint value plus the weight equals the right-hand value.
            assert current == self(arg) + jump
            if abs(current) > abs(max_point[1]):
                max_point  = (arg, current)
                if abs(current) > limit:
                    break
        return max_point

    def total_sign_jump(self):
        """Return the sum of all stored jump weights."""
        # Knot signatures normally have total jump zero after a full turn,
        # making this a useful structural check for computed examples.
        return sum([j[1] for j in sorted(self._jumps_counter.items())])

    def plot(self, *args, **kargs):
        """Delegate rendering and return the resulting Figure or Axes object."""
        return SignaturePloter.plot(self, *args, **kargs)


# ---------------------------------------------------------------------------
# Matplotlib and TikZ rendering
# ---------------------------------------------------------------------------

class SignaturePloter:
    """Render one or more signature functions as step functions.

    The historical spelling ``SignaturePloter`` is retained for compatibility.
    Supplying an existing axes with ``subplot=True`` lets composite layouts use
    ``plot`` without independently displaying or closing their figures.
    """

    @classmethod
    def plot_many(cls, *sf_list, save_path=None, title='', cols=None):
        """Arrange up to 36 signature functions in a shared-axis grid."""
        axes_num = len(sf_list)
        # Matplotlib cannot construct a useful zero-sized grid.
        if axes_num == 0:
            raise ValueError("At least one signature function is required.")

        if axes_num > 36:
            # Very large grids become unreadable. Keep the historical cap while
            # making truncation explicit to the caller.
            sf_list = sf_list[:36]
            axes_num = 36
            warnings.warn(
                "Too many signature functions were given. Only 36 are plotted.",
                stacklevel=2,
            )

        if cols is None:
            # A near-square default avoids producing one very long row.
            cols = math.ceil(math.sqrt(axes_num))
        elif cols <= 0 or int(cols) != cols:
            raise ValueError("The number of columns must be a positive integer.")
        cols = int(cols)
        rows = math.ceil(axes_num / cols)

        # squeeze=False guarantees a two-dimensional axes array even for a
        # single row or column.
        fig, axes_matrix = plt.subplots(
            rows,
            cols,
            squeeze=False,
            sharex='col',
            sharey='row',
            gridspec_kw={'hspace': 0, 'wspace': 0},
        )

        for i, sf in enumerate(sf_list):
            row, col = divmod(i, cols)
            # Each function's plot_title labels its own panel; title below
            # labels the complete grid.
            cls.plot(
                sf,
                subplot=True,
                ax=axes_matrix[row][col],
                title=sf.plot_title,
            )

        # Do not leave empty panels visible when the grid is not completely
        # filled by the requested functions.
        for i in range(axes_num, rows * cols):
            row, col = divmod(i, cols)
            axes_matrix[row][col].set_visible(False)

        fig.suptitle(title)
        fig.tight_layout()

        # Saved figures are closed by show_and_save, but returning the Figure
        # still lets callers inspect its axes and metadata.
        cls.show_and_save(fig, save_path)
        return fig

    @classmethod
    def plot_sum_of_two(cls, sf1, sf2, save_path=None, title=''):
        """Plot two functions, their sum, and an overlay in four panels."""
        # Reuse the same sum object in the standalone and overlay panels.
        sf = sf1 + sf2
        fig, axes_matrix = plt.subplots(2, 2, sharey=True, figsize=(10, 5))

        cls.plot(sf1, subplot=True, ax=axes_matrix[0][1])

        cls.plot(
            sf2,
            subplot=True,
            ax=axes_matrix[1][0],
            color='red',
            linestyle='dotted',
        )

        cls.plot(sf, subplot=True, ax=axes_matrix[0][0], color='black')

        cls.plot(sf1, subplot=True, ax=axes_matrix[1][1], alpha=0.3)

        cls.plot(
            sf2,
            subplot=True,
            ax=axes_matrix[1][1],
            color='red',
            alpha=0.3,
            linestyle='dotted',
        )

        cls.plot(
            sf,
            subplot=True,
            ax=axes_matrix[1][1],
            color='black',
            alpha=0.7,
        )

        fig.suptitle(title)
        fig.tight_layout()

        cls.show_and_save(fig, save_path)
        return fig

    @classmethod
    def plot(cls, sf, subplot=False, ax=None,
             save_path=None,
             title='',
             alpha=1,
             color='blue',
             linestyle='solid',
             special_point=None,
             special_label='',
             extraticks=None,
             ylabel=''):
        """Draw one function as horizontal constant-value segments.

        With ``subplot=True``, return the axes without showing or saving it.
        Otherwise apply the requested output policy and return the figure.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.figure

        segments = cls._step_segments(sf)
        if segments:
            # hlines accepts parallel collections of heights and endpoints, so
            # unzip the segment triples produced from the sparse counter.
            xmin, xmax, y = zip(*segments)
            ax.hlines(
                y,
                xmin,
                xmax,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
            )

        ax.set(ylabel=ylabel)
        ax.set(title=title)
        ax.set_xlim(0, 1)

        if special_point is not None:
            # Highlight an optional mathematically distinguished point while
            # retaining any exact tick locations requested by the caller.
            arg, val = special_point
            extraticks = extraticks or []
            ax.set_xticks(list(ax.get_xticks()) + extraticks)
            ext = sf.extremum()[1]
            ytext = ext/2 + 1/2
            xtext = arg + 1/5

            ax.annotate(special_label, xy=(arg, val), xytext=(xtext, ytext),
                        arrowprops=dict(facecolor='black', shrink=0.05,
                                        alpha=0.7, width=2),)
        if subplot:
            return ax

        cls.show_and_save(fig, save_path)
        return fig

    @staticmethod
    def show_and_save(fig, save_path):
        """Save ``fig`` as PNG when requested, otherwise display it."""
        if save_path is not None:
            # Normalize any supplied suffix to the documented PNG format.
            save_path = Path(save_path)
            save_path = save_path.with_suffix('.png')
            fig.savefig(save_path)
            plt.close(fig)
            return save_path

        # Matplotlib chooses suitable behavior for interactive sessions,
        # notebooks, and noninteractive backends such as the test backend.
        plt.show()
        return None

    @staticmethod
    def _step_segments(sf):
        """Return ``(left, right, value)`` triples covering ``[0, 1)``.

        Including the initial and final intervals makes the zero tails of a
        balanced signature visible rather than plotting only between jumps.
        """
        counter = sf.jumps_counter
        segments = []
        # Project normalization starts at zero immediately before the first
        # discontinuity.
        current_value = 0
        left = 0

        for jump_location, jump in sorted(counter.items()):
            # A jump at zero creates no positive-length initial segment.
            if jump_location > left:
                segments.append((left, jump_location, current_value))
            # The open interval to the right differs by twice the jump weight.
            current_value += 2 * jump
            left = jump_location

        # Complete the half-open fundamental domain after the last jump.
        if left < 1:
            segments.append((left, 1, current_value))

        return segments

    @staticmethod
    def step_function_data(sf):
        """Return every jump location and its right-hand function value.

        Evaluation at a jump gives the midpoint, so adding its weight selects
        the right-hand limit.
        """
        counter = sf.jumps_counter
        return [
            (jump_location, sf(jump_location) + jump)
            for jump_location, jump in sorted(counter.items())
        ]

    @classmethod
    def tikz_plot(cls, sf, save_as):
        """Write a standalone TikZ step plot and return its ``.tex`` path.

        Horizontal solid segments encode constant intervals. Dotted vertical
        segments show discontinuities without choosing either one-sided value.
        """
        # Always create TeX source regardless of the caller's original suffix.
        save_path = Path(save_as).with_suffix('.tex')
        lines = [
            r"\documentclass[tikz]{standalone}",
            r"\begin{document}",
            r"\begin{tikzpicture}[x=10cm,y=0.5cm]",
            r"\draw[->] (0,0) -- (1.05,0) node[right] {$\theta$};",
        ]

        for left, right, value in cls._step_segments(sf):
            # Compact decimals are preferable to long rational expansions in
            # drawing coordinates while remaining accurate for visualization.
            lines.append(
                "\\draw[thick] "
                f"({float(left):.12g},{float(value):.12g}) -- "
                f"({float(right):.12g},{float(value):.12g});"
            )

        # Dotted vertical segments make the discontinuity locations visible
        # without choosing either one-sided value as the value at the jump.
        for jump_location, jump in sorted(sf.jumps_counter.items()):
            midpoint = sf(jump_location)
            left_value = midpoint - jump
            right_value = midpoint + jump
            lines.append(
                "\\draw[densely dotted] "
                f"({float(jump_location):.12g},{float(left_value):.12g}) -- "
                f"({float(jump_location):.12g},{float(right_value):.12g});"
            )

        lines.extend([
            r"\end{tikzpicture}",
            r"\end{document}",
        ])
        # A final newline keeps the generated file friendly to Unix tools and
        # version-control diffs.
        save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return save_path
