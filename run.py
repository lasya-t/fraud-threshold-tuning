"""Train a fraud model on synthetic device orders, then choose how to act on its scores."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split

from threshold import CostModel, best_threshold, sweep, value_aware_flags


def make_orders(n=60_000, seed=0):
    """Orders with a device value and features that partly reveal fraud."""
    rng = np.random.default_rng(seed)
    value = rng.choice([30, 199, 499, 799, 1099, 1299], n, p=[0.25, 0.2, 0.15, 0.2, 0.12, 0.08]).astype(float)
    p_fraud = 0.004 + 0.02 * (value >= 799)                  # fraudsters prefer expensive phones
    y = (rng.random(n) < p_fraud).astype(int)
    account_age = np.where(y == 1, rng.exponential(5, n), rng.exponential(300, n))
    ship_mismatch = rng.random(n) < np.where(y == 1, 0.7, 0.05)
    night = rng.random(n) < np.where(y == 1, 0.4, 0.15)
    devices_in_order = rng.poisson(np.where(y == 1, 1.5, 0.2)) + 1
    noise = rng.normal(size=(n, 4))
    X = np.column_stack([np.log1p(value), np.log1p(account_age), ship_mismatch, night, devices_in_order, noise])
    # blur the signal so the model isn't perfect
    X[:, 1] += rng.normal(0, 1.5, n)
    return X, y, value


X, y, value = make_orders()
Xtr, Xte, ytr, yte, vtr, vte = train_test_split(X, y, value, test_size=0.4, stratify=y, random_state=0)
Xtr, Xcal, ytr, ycal, vtr, vcal = train_test_split(Xtr, ytr, vtr, test_size=0.3, stratify=ytr, random_state=0)

model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(Xcal, ycal)
p_cal = cal.predict_proba(Xcal)[:, 1]
p_te = cal.predict_proba(Xte)[:, 1]

cm = CostModel()
print(f"test orders {len(yte):,}, fraud {yte.sum()} ({yte.mean():.2%}), PR-AUC {average_precision_score(yte, p_te):.3f}\n")


def report(label, flagged):
    c = cm.per_order(yte, vte, flagged).sum()
    caught = (flagged & (yte == 1)).sum()
    print(f"{label:34s} cost ${c:>10,.0f}  flagged {flagged.sum():5d}  caught {caught:3d}/{yte.sum()}  "
          f"false alarms {(flagged & (yte == 0)).sum():5d}")
    return c


# thresholds are chosen on the calibration split, then applied to the test split
f1s = {t: f1_score(ycal, p_cal >= t) for t in np.linspace(0.01, 0.99, 99)}
t_f1 = max(f1s, key=f1s.get)
t_cost = best_threshold(cm, ycal, vcal, p_cal)["threshold"]

no_model = report("no screening (approve all)", np.zeros(len(yte), bool))
report("default threshold 0.5", p_te >= 0.5)
report_cost_f1 = report(f"best-F1 threshold {t_f1:.2f}", p_te >= t_f1)
c_cost = report(f"min-cost threshold {t_cost:.3f}", p_te >= t_cost)
c_value = report("value-aware rule (per order)", value_aware_flags(cm, vte, p_te))
print(f"\nmin-cost threshold saves ${no_model - c_cost:,.0f} vs no screening "
      f"and ${report_cost_f1 - c_cost:,.0f} vs the best-F1 threshold")
print(f"value-aware rule vs min-cost threshold: ${c_value - c_cost:+,.0f}")

rows = sweep(cm, yte, vte, p_te)
ts = [r["threshold"] for r in rows]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ts, [r["cost"] / 1000 for r in rows])
for t, lab in [(0.5, "0.5"), (t_f1, "best F1"), (t_cost, "min cost")]:
    ax.axvline(t, ls="--", lw=1, c="gray")
    ax.text(t, ax.get_ylim()[1] * 0.95, lab, rotation=90, va="top", fontsize=8)
ax.set(xlabel="threshold", ylabel="total cost ($k)", title="Business cost by decision threshold", xscale="symlog", xlim=(0, 1))
fig.tight_layout()
fig.savefig("cost_curve.png", dpi=110)
print("saved cost_curve.png")
