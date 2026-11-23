"""Pick a fraud decision threshold that minimizes business cost.

Each scored order ends up in one of four buckets:

    fraud,  flagged      -> caught: pay a review cost, recover most of the loss
    fraud,  not flagged  -> missed: lose the device value + chargeback fee
    legit,  flagged      -> false alarm: review cost + some customers abandon the order
    legit,  not flagged  -> nothing

Costs depend on the order's value, so the cost of a threshold is summed
order by order instead of using one fixed cost per error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    review_cost: float = 8.0          # analyst time per flagged order
    chargeback_fee: float = 25.0      # bank fee on a missed fraud
    recovery_rate: float = 0.9        # share of a caught fraud's value saved
    abandon_rate: float = 0.15        # legit customers who walk away when held for review
    margin: float = 0.25              # profit margin lost when a legit order is abandoned

    def per_order(self, y: np.ndarray, value: np.ndarray, flagged: np.ndarray) -> np.ndarray:
        caught = (y == 1) & flagged
        missed = (y == 1) & ~flagged
        false_alarm = (y == 0) & flagged
        cost = np.zeros(len(y))
        cost[caught] = self.review_cost + (1 - self.recovery_rate) * value[caught]
        cost[missed] = value[missed] + self.chargeback_fee
        cost[false_alarm] = self.review_cost + self.abandon_rate * self.margin * value[false_alarm]
        return cost

    def total(self, y, value, scores, threshold) -> float:
        return float(self.per_order(y, value, scores >= threshold).sum())


def sweep(cost_model: CostModel, y, value, scores, thresholds=None):
    """Total cost and confusion counts across thresholds."""
    thresholds = np.linspace(0, 1, 201) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        flagged = scores >= t
        rows.append({
            "threshold": float(t),
            "cost": float(cost_model.per_order(y, value, flagged).sum()),
            "flagged": int(flagged.sum()),
            "caught": int((flagged & (y == 1)).sum()),
            "missed": int((~flagged & (y == 1)).sum()),
            "false_alarms": int((flagged & (y == 0)).sum()),
        })
    return rows


def best_threshold(cost_model: CostModel, y, value, scores):
    rows = sweep(cost_model, y, value, scores)
    return min(rows, key=lambda r: r["cost"])


def value_aware_flags(cost_model: CostModel, value, prob):
    """Flag when the expected cost of letting an order through exceeds the expected cost of reviewing it.

    Needs calibrated probabilities. The break-even probability is lower for
    expensive orders, so a $1,200 phone gets reviewed at a lower risk score
    than a $30 case.
    """
    let_through = prob * (value + cost_model.chargeback_fee)
    review = (
        cost_model.review_cost
        + prob * (1 - cost_model.recovery_rate) * value
        + (1 - prob) * cost_model.abandon_rate * cost_model.margin * value
    )
    return let_through > review
