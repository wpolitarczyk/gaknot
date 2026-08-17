"""Unit and integration tests for sparse signature functions.

Most tests construct ``SignatureFunction`` directly from small exact counters.
That keeps expected values independent of the torus-knot algorithms and makes
midpoint, algebraic, transformation, and caching behavior visible. Plot tests
use Matplotlib's noninteractive backend and pytest temporary directories, so
they neither open windows nor leave artifacts in the repository.

The final two tests are intentionally higher-level bridges: the trefoil fixture
checks the jump convention against a known knot, and the generalized-knot test
checks delegation through ``GeneralizedAlgebraicKnot.signature()``.
"""

from collections import Counter

import matplotlib
import pytest
from sage.all import QQ

# Select the backend before importing pyplot or the package module that imports
# it. ``force=True`` also makes this deterministic if Sage selected a backend.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from gaknot import GeneralizedAlgebraicKnot, SignatureFunction, SignaturePloter
from gaknot.invariants.LT_signature import LT_signature_torus_knot


# ---------------------------------------------------------------------------
# Construction, domain invariants, and defensive exposure
# ---------------------------------------------------------------------------

def test_signature_function_construction_aggregates_duplicate_jumps():
    # Two contributions at 1/4 add to 3, while opposite contributions at 1/2
    # cancel and should be removed from the sparse representation entirely.
    quarter = QQ(1) / 4
    half = QQ(1) / 2

    signature = SignatureFunction(values=[
        (quarter, 1),
        (quarter, 2),
        (half, -1),
        (half, 1),
    ])

    assert signature.jumps_counter == Counter({quarter: 3})


def test_signature_function_copies_input_counter():
    # Constructor ownership is independent of the public defensive-copy
    # property tested below: mutating the original input must also be harmless.
    quarter = QQ(1) / 4
    source_counter = Counter({quarter: 1})

    signature = SignatureFunction(counter=source_counter)
    source_counter[quarter] = 5

    assert signature.jumps_counter == Counter({quarter: 1})


@pytest.mark.parametrize("jump_location", [
    -QQ(1) / 4,
    QQ(1),
    QQ(5) / 4,
])
@pytest.mark.parametrize("input_name", ["values", "counter"])
def test_signature_function_rejects_jump_locations_outside_domain(
    jump_location,
    input_name,
):
    # Exercise both supported construction paths at the lower boundary, the
    # excluded endpoint 1, and a location beyond one full turn.
    input_value = (
        [(jump_location, 1)]
        if input_name == "values"
        else Counter({jump_location: 1})
    )

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        SignatureFunction(**{input_name: input_value})


def test_signature_function_jump_counter_is_defensively_copied():
    # Alter both an existing coefficient and the set of keys in the exported
    # Counter. Neither change may reach the private counter or its caches.
    quarter = QQ(1) / 4
    half = QQ(1) / 2
    signature = SignatureFunction(values=[(quarter, 1)])

    exported_counter = signature.jumps_counter
    exported_counter[quarter] = 5
    exported_counter[half] = -1

    assert signature.jumps_counter == Counter({quarter: 1})
    # The unchanged value after the original jump proves evaluation still uses
    # the matching immutable counter/cumulative-sum snapshot.
    assert signature(half) == 2


# ---------------------------------------------------------------------------
# Evaluation and algebra
# ---------------------------------------------------------------------------

# The first six rows cover the interval before, at, between, at, and after the
# two jumps. The last three verify reduction modulo one in both directions.
@pytest.mark.parametrize("argument, expected", [
    (QQ(0), 0),
    (QQ(1) / 8, 0),
    (QQ(1) / 4, 1),
    (QQ(1) / 2, 2),
    (QQ(3) / 4, 1),
    (QQ(7) / 8, 0),
    (QQ(1), 0),
    (-QQ(1) / 2, 2),
    (QQ(3) / 2, 2),
])
def test_signature_function_evaluation_and_periodicity(argument, expected):
    # Crossing the +1 jump raises the value by 2; crossing the -1 jump restores
    # zero. At either jump the expected value is the midpoint of the limits.
    signature = SignatureFunction(values=[
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -1),
    ])

    assert signature(argument) == expected


def test_signature_function_equality_uses_complete_jump_counter():
    # same_signature reaches the same normalized Counter through a different
    # order and a duplicate location, while different_signature omits a jump.
    quarter = QQ(1) / 4
    half = QQ(1) / 2

    signature = SignatureFunction(values=[(quarter, 1), (half, -1)])
    same_signature = SignatureFunction(
        values=[(half, -2), (quarter, 1), (half, 1)]
    )
    different_signature = SignatureFunction(values=[(quarter, 1)])

    assert signature == same_signature
    assert signature != different_signature


def test_signature_function_arithmetic():
    # Overlap at 1/2 checks coefficient-wise combination rather than simple
    # concatenation of disjoint counters.
    quarter = QQ(1) / 4
    half = QQ(1) / 2
    three_quarters = QQ(3) / 4
    left = SignatureFunction(values=[(quarter, 1), (half, -1)])
    right = SignatureFunction(values=[(half, 2), (three_quarters, -1)])

    assert (left + right).jumps_counter == Counter({
        quarter: 1,
        half: 1,
        three_quarters: -1,
    })
    assert (left - right).jumps_counter == Counter({
        quarter: 1,
        half: -3,
        three_quarters: 1,
    })
    assert (-left).jumps_counter == Counter({quarter: -1, half: 1})
    assert (3 * left).jumps_counter == Counter({quarter: 3, half: -3})
    assert (left * 3).jumps_counter == Counter({quarter: 3, half: -3})
    # Multiplication by zero must be normalized to the sparse zero function.
    assert (0 * left).is_zero_everywhere()

    # All operations are functional: neither operand may be mutated in place.
    assert left.jumps_counter == Counter({quarter: 1, half: -1})
    assert right.jumps_counter == Counter({half: 2, three_quarters: -1})


def test_signature_function_addition_combines_plot_titles():
    # Presentation metadata does not affect equality, but addition should keep
    # useful labels for plots in every titled/untitled combination.
    quarter = QQ(1) / 4
    left = SignatureFunction(values=[(quarter, 1)], plot_title="left")
    right = SignatureFunction(values=[(quarter, -1)], plot_title="right")
    untitled = SignatureFunction(values=[(quarter, 2)])

    assert (left + right).plot_title == "left + right"
    assert (left + untitled).plot_title == "left"
    assert (untitled + right).plot_title == "right"


# ---------------------------------------------------------------------------
# Argument transformations and summary helpers
# ---------------------------------------------------------------------------

# A quarter-turn moves the two jumps to {0, 1/2} in either direction, with
# their weights exchanged between those locations as dictated by orientation.
@pytest.mark.parametrize("operator, expected_counter", [
    (
        lambda signature: signature >> (QQ(1) / 4),
        Counter({QQ(1) / 2: 1, QQ(0): -1}),
    ),
    (
        lambda signature: signature << (QQ(1) / 4),
        Counter({QQ(0): 1, QQ(1) / 2: -1}),
    ),
])
def test_signature_function_shifts(operator, expected_counter):
    signature = SignatureFunction(values=[
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -1),
    ])

    assert operator(signature).jumps_counter == expected_counter
    # Applying opposite rotations is an exact inverse operation.
    assert ((signature >> (QQ(1) / 4)) << (QQ(1) / 4)) == signature


def test_signature_function_double_cover():
    # Pullback by theta -> 2*theta gives every original jump two preimages, one
    # in each half of the unit interval.
    signature = SignatureFunction(values=[
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -1),
    ])

    assert signature.double_cover().jumps_counter == Counter({
        QQ(1) / 8: 1,
        QQ(3) / 8: -1,
        QQ(5) / 8: 1,
        QQ(7) / 8: -1,
    })


def test_signature_function_square_root_branches():
    # Give every quadrant a distinct weight so branch selection and rescaling
    # can be detected independently rather than hidden by symmetry.
    signature = SignatureFunction(values=[
        (QQ(1) / 8, 1),
        (QQ(3) / 8, -2),
        (QQ(5) / 8, 3),
        (QQ(7) / 8, -4),
    ])

    assert signature.square_root().jumps_counter == Counter({
        QQ(1) / 4: 1,
        QQ(3) / 4: -2,
    })
    assert signature.minus_square_root().jumps_counter == Counter({
        QQ(1) / 4: 3,
        QQ(3) / 4: -4,
    })


@pytest.mark.parametrize("limit, expected", [
    # Without a limit, the right-hand value 4 after the second jump is maximal.
    (None, (QQ(1) / 4, 4)),
    # A limit of 1 permits early termination at the first right-hand value 2.
    (1, (QQ(1) / 8, 2)),
])
def test_signature_function_extremum(limit, expected):
    signature = SignatureFunction(values=[
        (QQ(1) / 8, 1),
        (QQ(1) / 4, 1),
        (QQ(1) / 2, -3),
        (QQ(3) / 4, 1),
    ])

    result = signature.extremum() if limit is None else signature.extremum(limit)

    assert result == expected


def test_signature_function_zero_and_total_jump_helpers():
    # The balanced function is nonzero on part of the circle even though its
    # total jump vanishes, distinguishing the two predicates.
    zero_signature = SignatureFunction()
    balanced_signature = SignatureFunction(values=[
        (QQ(1) / 4, 2),
        (QQ(3) / 4, -2),
    ])

    assert zero_signature.is_zero_everywhere()
    assert zero_signature.total_sign_jump() == 0
    assert not balanced_signature.is_zero_everywhere()
    assert balanced_signature.total_sign_jump() == 0


def test_signature_function_text_representations_are_sorted():
    # Supply jumps out of order to ensure output follows mathematical location,
    # not input order.
    signature = SignatureFunction(values=[
        (QQ(3) / 4, -1),
        (QQ(1) / 4, 1),
    ])

    assert str(signature) == "1/4: 1\n3/4: -1\n"
    assert repr(signature) == "1/4: 1, 3/4: -1."


def test_zero_signature_has_meaningful_representation():
    # This is a regression for the former representation ".".
    assert repr(SignatureFunction()) == "SignatureFunction()"


# ---------------------------------------------------------------------------
# Matplotlib and TikZ rendering
# ---------------------------------------------------------------------------

def test_signature_plot_draws_complete_step_function_on_subplot():
    # The expected horizontal pieces are 0 on [0,1/4), 2 on [1/4,3/4),
    # and 0 on [3/4,1). Including both tails is part of the plotting contract.
    signature = SignatureFunction(values=[
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -1),
    ])
    figure, axis = plt.subplots()

    returned_axis = signature.plot(
        subplot=True,
        ax=axis,
        title="example",
        ylabel="signature",
    )

    assert returned_axis is axis
    # subplot=True must reuse and return the caller-owned axes.
    assert axis.get_title() == "example"
    assert axis.get_ylabel() == "signature"
    segments = axis.collections[0].get_segments()
    assert len(segments) == 3
    assert [segment[0][1] for segment in segments] == [0, 2, 0]
    assert [segment[0][0] for segment in segments] == pytest.approx([
        0,
        0.25,
        0.75,
    ])
    assert [segment[1][0] for segment in segments] == pytest.approx([
        0.25,
        0.75,
        1,
    ])
    plt.close(figure)


def test_signature_plot_saves_requested_png_without_temporary_file(
    tmp_path,
    monkeypatch,
):
    # Work inside tmp_path so this regression detects the historical tmp.png
    # side effect without risking an artifact in the repository.
    monkeypatch.chdir(tmp_path)
    signature = SignatureFunction(values=[(QQ(1) / 4, 1)])

    figure = signature.plot(save_path=tmp_path / "signature")

    assert len(figure.axes) == 1
    assert (tmp_path / "signature.png").is_file()
    # Saving should not invoke the old save-close-reopen workflow.
    assert not (tmp_path / "tmp.png").exists()


def test_signature_plot_many_builds_stable_grid_and_hides_unused_axis(tmp_path):
    # Three functions in two columns require a 2x2 grid with one unused panel.
    signatures = [
        SignatureFunction(values=[(QQ(1) / 4, value)])
        for value in (1, 2, 3)
    ]

    figure = SignaturePloter.plot_many(
        *signatures,
        cols=2,
        save_path=tmp_path / "many",
    )

    assert len(figure.axes) == 4
    assert all(axis.get_visible() for axis in figure.axes[:3])
    assert not figure.axes[3].get_visible()
    assert (tmp_path / "many.png").is_file()


def test_signature_plot_many_limits_output_to_36_functions(tmp_path):
    # The 37th function must be discarded by slicing, not selected as the sole
    # iterable as happened in the former sf_list[36] implementation.
    signature = SignatureFunction(values=[(QQ(1) / 4, 1)])

    with pytest.warns(UserWarning, match="Only 36 are plotted"):
        figure = SignaturePloter.plot_many(
            *([signature] * 37),
            cols=6,
            save_path=tmp_path / "limited",
        )

    assert len(figure.axes) == 36
    assert (tmp_path / "limited.png").is_file()


@pytest.mark.parametrize("signatures, cols, match", [
    # Empty input cannot define grid dimensions.
    ([], None, "At least one"),
    # Zero, negative, and fractional column counts are invalid layouts.
    ([SignatureFunction()], 0, "positive integer"),
    ([SignatureFunction()], -1, "positive integer"),
    ([SignatureFunction()], 1.5, "positive integer"),
])
def test_signature_plot_many_rejects_invalid_layout(signatures, cols, match):
    with pytest.raises(ValueError, match=match):
        SignaturePloter.plot_many(*signatures, cols=cols)


def test_signature_plot_sum_of_two_builds_four_panels(tmp_path):
    # The layout contains left, right, sum, and a combined overlay panel.
    left = SignatureFunction(values=[(QQ(1) / 4, 1)])
    right = SignatureFunction(values=[(QQ(3) / 4, -1)])

    figure = SignaturePloter.plot_sum_of_two(
        left,
        right,
        title="sum",
        save_path=tmp_path / "sum",
    )

    assert len(figure.axes) == 4
    assert figure._suptitle.get_text() == "sum"
    assert (tmp_path / "sum.png").is_file()


def test_signature_step_function_data_returns_right_hand_values():
    # Midpoint values at the jumps are 1 and 1; adding the corresponding +1
    # and -1 weights yields right-hand values 2 and 0.
    signature = SignatureFunction(values=[
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -1),
    ])

    assert SignaturePloter.step_function_data(signature) == [
        (QQ(1) / 4, 2),
        (QQ(3) / 4, 0),
    ]


def test_signature_tikz_plot_writes_standalone_step_function(tmp_path):
    # Inspect source text instead of requiring a TeX distribution in the unit
    # test environment. Matplotlib plotting is verified separately above.
    signature = SignatureFunction(values=[
        (QQ(1) / 4, 1),
        (QQ(3) / 4, -1),
    ])

    output_path = SignaturePloter.tikz_plot(signature, tmp_path / "signature")
    output = output_path.read_text(encoding="utf-8")

    assert output_path == tmp_path / "signature.tex"
    # Check the standalone wrapper, one characteristic horizontal interval,
    # and one discontinuity marker for each jump.
    assert r"\documentclass[tikz]{standalone}" in output
    assert r"\draw[thick] (0.25,2) -- (0.75,2);" in output
    assert output.count(r"\draw[densely dotted]") == 2


# ---------------------------------------------------------------------------
# Knot-level integration checks
# ---------------------------------------------------------------------------

def test_torus_knot_signature_values():
    # T(2,3) is the right-handed trefoil. Its jump weights are -1 at 1/6 and
    # +1 at 5/6. These independently known data test the abstract evaluation
    # convention against a concrete knot calculation.
    sig = LT_signature_torus_knot(2, 3)

    # Before 1/6: 0
    assert sig(QQ(1) / 10) == 0
    # At 1/6: -1
    assert sig(QQ(1) / 6) == -1
    # Between 1/6 and 5/6: 2 * (-1) = -2
    assert sig(QQ(1) / 2) == -2
    # At 5/6: 2*(-1) + (+1) = -1
    assert sig(QQ(5) / 6) == -1
    # After 5/6: 2*(-1) + 2*(1) = 0
    assert sig(QQ(9) / 10) == 0


def test_generalized_knot_signature_dispatches_exact_signed_sum():
    # Use a noncancelling difference: a broken dispatcher that always returned
    # the zero function would pass the former K # -K smoke test.
    knot = GeneralizedAlgebraicKnot([
        (1, [(2, 3)]),
        (-1, [(2, 5)]),
    ])
    # Assemble the expected function through elementary public constructors,
    # independently of GeneralizedAlgebraicKnot.signature().
    expected = LT_signature_torus_knot(2, 3) - LT_signature_torus_knot(2, 5)

    signature = knot.signature()

    assert signature == expected
    # Make the nonzero nature of the chosen fixture explicit in failure output.
    assert not signature.is_zero_everywhere()
