import numpy as np

from threshold import CostModel, best_threshold, value_aware_flags

cm = CostModel(review_cost=10, chargeback_fee=20, recovery_rate=1.0, abandon_rate=0.0)


def test_per_order_costs():
    y = np.array([1, 1, 0, 0])
    value = np.array([500.0, 500.0, 100.0, 100.0])
    flagged = np.array([True, False, True, False])
    np.testing.assert_allclose(cm.per_order(y, value, flagged), [10, 520, 10, 0])


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    best = best_threshold(cm, y, np.full(5, 300.0), scores)
    assert 0.3 < best["threshold"] <= 0.8
    assert best["missed"] == 0 and best["false_alarms"] == 0


def test_value_aware_reviews_expensive_orders_at_lower_risk():
    m = CostModel()
    prob = np.full(2, 0.06)  # break-even is ~4% for a $1,299 phone, ~16% for a $30 case
    flags = value_aware_flags(m, np.array([30.0, 1299.0]), prob)
    assert flags.tolist() == [False, True]
