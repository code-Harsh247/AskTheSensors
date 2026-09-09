"""Every metric is checked against a fixture whose expected value was worked
out by hand, so a metric cannot pass by agreeing with itself."""

import math

import pytest

from ats.eval import metrics

# gold = [A, A, B, B, C, C]
# pred = [A, B, B, B, C, A]
#   A: tp=1 fp=1 fn=1 -> P=1/2  R=1/2  F1=0.5
#   B: tp=2 fp=1 fn=0 -> P=2/3  R=1    F1=0.8
#   C: tp=1 fp=0 fn=1 -> P=1    R=1/2  F1=2/3
TOY_GOLD = ["A", "A", "B", "B", "C", "C"]
TOY_PRED = ["A", "B", "B", "B", "C", "A"]
TOY_LABELS = ["A", "B", "C"]


def test_accuracy_toy():
    assert metrics.accuracy(TOY_PRED, TOY_GOLD) == pytest.approx(4 / 6)


def test_confusion_matrix_toy():
    assert metrics.confusion_matrix(TOY_PRED, TOY_GOLD, TOY_LABELS) == [
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ]


def test_per_class_prf_toy():
    scores = metrics.per_class_prf(TOY_PRED, TOY_GOLD, TOY_LABELS)
    assert scores["A"]["precision"] == pytest.approx(0.5)
    assert scores["A"]["recall"] == pytest.approx(0.5)
    assert scores["B"]["precision"] == pytest.approx(2 / 3)
    assert scores["B"]["recall"] == pytest.approx(1.0)
    assert scores["B"]["f1"] == pytest.approx(0.8)
    assert scores["C"]["f1"] == pytest.approx(2 / 3)
    assert scores["C"]["support"] == 2


def test_macro_f1_toy():
    assert metrics.macro_f1(TOY_PRED, TOY_GOLD, TOY_LABELS) == pytest.approx(
        (0.5 + 0.8 + 2 / 3) / 3
    )


def test_balanced_accuracy_toy():
    assert metrics.balanced_accuracy(TOY_PRED, TOY_GOLD, TOY_LABELS) == pytest.approx(
        (0.5 + 1.0 + 0.5) / 3
    )


def test_macro_f1_differs_from_accuracy_under_imbalance():
    """The reason macro-F1 is reported at all: a majority-class predictor
    scores well on accuracy and badly on macro-F1."""
    gold = ["SITTING"] * 9 + ["RUNNING"]
    pred = ["SITTING"] * 10
    assert metrics.accuracy(pred, gold) == pytest.approx(0.9)
    assert metrics.macro_f1(pred, gold, ["SITTING", "RUNNING"]) == pytest.approx(
        (2 * 0.9 * 1.0 / 1.9 + 0.0) / 2
    )


# gold = [Yes, Yes, No, No, No]
# pred = [Yes, No,  Yes, No, No]  ->  tp=1 fn=1 fp=1 tn=2
def test_binary_verification_hand_counts():
    result = metrics.binary_verification(
        ["Yes", "No", "Yes", "No", "No"], ["Yes", "Yes", "No", "No", "No"]
    )
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 2)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(0.6)


def test_mae_hand_computed():
    # errors 10, 20, 0 -> mean 10
    assert metrics.mae([100, 200, 300], [110, 180, 300]) == pytest.approx(10.0)


def test_mape_hand_computed():
    # 10/110 + 20/180 + 0 = 0.2020202 over 3 pairs
    assert metrics.mape([100, 200, 300], [110, 180, 300]) == pytest.approx(
        (10 / 110 + 20 / 180) / 3
    )


def test_mape_skips_zero_gold():
    # only the 50-vs-100 pair is usable
    assert metrics.mape([10, 50], [0, 100]) == pytest.approx(0.5)


def test_within_tolerance_relative():
    # 10% of 650 is 65; an error of 50 is inside it, 70 is not
    assert metrics.within_tolerance(700, 650, abs_tol=None, rel_tol=0.10)
    assert not metrics.within_tolerance(720, 650, abs_tol=None, rel_tol=0.10)


def test_within_tolerance_takes_the_larger_of_abs_and_rel():
    # 10% of 20 is 2, but an absolute floor of 30 dominates
    assert metrics.within_tolerance(45, 20, abs_tol=30, rel_tol=0.10)


def test_tolerance_accuracy_hand_computed():
    assert metrics.tolerance_accuracy(
        [700, 720], [650, 650], abs_tol=None, rel_tol=0.10
    ) == pytest.approx(0.5)


def test_interval_iou_hand_computed():
    # (10,20) vs (15,30): overlap 5, union 20 -> 0.25
    assert metrics.interval_iou((10, 20), (15, 30)) == pytest.approx(0.25)


def test_interval_iou_identical_and_disjoint():
    assert metrics.interval_iou((0, 10), (0, 10)) == pytest.approx(1.0)
    assert metrics.interval_iou((0, 10), (20, 30)) == pytest.approx(0.0)


def test_temporal_prf_hand_computed():
    # pred [(0,10),(20,30)] vs gold [(5,25)]: overlap 5+5=10,
    # pred length 20, gold length 20 -> P=R=F1=0.5
    result = metrics.temporal_prf([(0, 10), (20, 30)], [(5, 25)])
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_matched_mean_iou_penalises_missing_intervals():
    # one gold interval matched perfectly, the other unmatched -> (1+0)/2
    assert metrics.matched_mean_iou([(0, 10)], [(0, 10), (50, 60)]) == pytest.approx(0.5)


def test_grounding_accuracy_at_threshold():
    preds = [[(0, 10)], [(0, 10)]]
    golds = [[(0, 10)], [(8, 30)]]  # second pair IoU = 2/30
    assert metrics.grounding_accuracy(preds, golds, threshold=0.5) == pytest.approx(0.5)


def test_channels_match_treats_all_as_superset():
    assert metrics.channels_match(["all"], ["acc_x", "acc_y"])
    assert metrics.channels_match(["acc_x"], ["acc_x"])
    assert not metrics.channels_match(["acc_x"], ["gyro_x"])


def test_is_grounded_requires_correct_answer_interval_and_modality():
    answer = {
        "cited_intervals": [[0, 10]],
        "modality": "both",
        "channels": ["all"],
    }
    gold = {"cited_intervals": [[0, 10]], "modality": "both", "channels": ["all"]}

    assert metrics.is_grounded_and_correct(answer, gold, answer_correct=True)
    assert not metrics.is_grounded_and_correct(answer, gold, answer_correct=False)

    wrong_interval = dict(answer, cited_intervals=[[500, 510]])
    assert not metrics.is_grounded_and_correct(wrong_interval, gold, answer_correct=True)

    wrong_modality = dict(answer, modality="accelerometer")
    assert not metrics.is_grounded_and_correct(wrong_modality, gold, answer_correct=True)


def test_macro_average_ignores_nan():
    assert metrics.macro_average([0.5, 1.0, math.nan]) == pytest.approx(0.75)


def test_metrics_raise_on_empty_input():
    with pytest.raises(ValueError):
        metrics.accuracy([], [])
    with pytest.raises(ValueError):
        metrics.mae([], [])


def test_normalize_categorical_is_case_and_punctuation_insensitive():
    assert metrics.categorical_match("Walking", "  walking. ")
    assert not metrics.categorical_match("Walking", "Running")
