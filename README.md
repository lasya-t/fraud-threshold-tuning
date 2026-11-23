# Fraud Threshold Tuning by Business Cost

A fraud model outputs a score, and someone still has to decide **at what score an order gets held for review**. The default of 0.5, or the threshold that maximizes F1, ignores what each mistake costs. This project picks the threshold that minimizes dollars lost instead.

## Cost model

Each order's cost depends on its value:

| Outcome | Cost |
|---|---|
| Fraud, flagged (caught) | $8 review + 10% of the device value not recovered |
| Fraud, not flagged (missed) | full device value + $25 chargeback |
| Legit, flagged (false alarm) | $8 review + 15% of customers abandon the order × 25% margin |
| Legit, not flagged | $0 |

All of these are parameters in `CostModel`, so they can be swapped for real numbers.

## Setup

- 60,000 synthetic device orders worth $30–$1,299, with 1.2% fraud. Fraudsters prefer expensive phones, new accounts, mismatched shipping states and multi-device orders.
- Gradient boosting model, calibrated with isotonic regression on a separate split.
- **Thresholds are chosen on the calibration split and evaluated on a separate test split**, so the comparison is fair.

## Results (test split: 24,000 orders, 280 fraud, PR-AUC 0.70)

| Decision rule | Total cost | Flagged | Fraud caught | False alarms |
|---|---|---|---|---|
| No screening | $242,402 | 0 | 0 / 280 | 0 |
| Default threshold 0.5 | $131,329 | 168 | 141 / 280 | 27 |
| Best-F1 threshold (0.26) | $91,556 | 280 | 193 / 280 | 87 |
| **Min-cost threshold (0.035)** | **$67,698** | 658 | 237 / 280 | 421 |
| Value-aware rule (per order) | $71,237 | 572 | 228 / 280 | 344 |

## What this shows

- **The cost-optimal threshold is far lower than 0.5 (0.035).** A missed $1,099 phone costs about 100× more than a false alarm, so it's worth reviewing 421 good orders to catch 96 more frauds than the 0.5 cutoff does.
- **Cost tuning saved about $24K over the best-F1 threshold** on this test set. F1 weighs a missed fraud and a false alarm equally, which they aren't.
- **The value-aware rule** sets a separate break-even probability for each order: about 4% for a $1,299 phone vs about 16% for a $30 case. In theory it's optimal, but it came in about $3.5K *worse* than the single tuned threshold. It relies on the probabilities being exactly right, and isotonic calibration fitted on about 200 fraud cases isn't precise enough at very low scores. With more calibration data or smoother calibration, this could flip.

`cost_curve.png` plots total cost against threshold, with the three cutoffs marked.

## How to run

```bash
pip install -r requirements.txt
PYTHONPATH=src python run.py
pytest
```

## Next steps
- Add a daily review-capacity limit, so the queue can only hold N orders
- Use the telecom-orders-generator dataset instead of this simplified data
