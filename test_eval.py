import numpy as np
import pytest

from wheatkit.eval import average_domain_accuracy, box_iou, match, mean_iou, metrics_by_group, regression_metrics


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-9)


def test_match_counts():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    pred = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
    assert match(pred, gt) == (1, 1, 1)
    assert match(np.zeros((0, 4)), gt) == (0, 0, 2)
    assert match(pred, np.zeros((0, 4))) == (0, 2, 0)


def test_one_prediction_cannot_match_two_heads():
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10.5]])
    pred = np.array([[0, 0, 10, 10]])
    assert match(pred, gt) == (1, 0, 1)


def test_average_domain_accuracy_weights_domains_equally():
    gt = [np.array([[0, 0, 10, 10]])] * 3
    preds = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]), np.zeros((0, 4))]
    ada, per = average_domain_accuracy(preds, gt, ["big", "big", "small"])
    assert per == {"big": 1.0, "small": 0.0}
    assert ada == pytest.approx(0.5)


def test_empty_image_both_empty_is_perfect():
    ada, _ = average_domain_accuracy([np.zeros((0, 4))], [np.zeros((0, 4))], ["d"])
    assert ada == 1.0


def test_regression_metrics():
    m = regression_metrics([10, 20, 30], [12, 18, 33])
    assert m["mae"] == pytest.approx(7 / 3)
    assert m["bias"] == pytest.approx(1.0)
    assert m["rrmse"] == pytest.approx(m["rmse"] / 20)


def test_metrics_by_group_has_summary_rows():
    df = metrics_by_group([1, 2, 3, 4], [1, 2, 3, 5], ["a", "a", "b", "b"])
    assert set(df.group) == {"a", "b", "__all__", "__mean_of_groups__"}


def test_mean_iou():
    gt = np.array([[0, 0], [1, 1]])
    pred = np.array([[0, 1], [1, 1]])
    out = mean_iou([pred], [gt], 2, ["bg", "head"])
    assert out["iou_bg"] == pytest.approx(0.5)
    assert out["iou_head"] == pytest.approx(2 / 3)
