# Can Machine Learning Predict the Stock Market?

### A controlled comparison of a linear baseline vs. a nonlinear ensemble for next-day NVDA direction

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-orange)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

**Author:** Roshia Dutta · University of Washington

---

## Research Question

> Does a more flexible model (Random Forest) predict next-day stock direction better than a simple linear model (Logistic Regression), and does either beat a trivial baseline?

---

## Why This Is Worth Studying

Stock prediction is a well-studied problem, so the goal here is **not** to claim a market-beating model. The value is in **how carefully the question is tested**.

- **Tests a real theory.** The Efficient Market Hypothesis (Fama, 1970) says short-term price moves should be very hard to predict. This project gives an empirical check on that claim for a high-profile, AI-driven stock (NVDA).
- **Isolates model complexity.** Both models get the *same* data, features, and split. The only thing that changes is the model, so any difference comes from added complexity.
- **Uses a "do-nothing" baseline.** A majority-class `DummyClassifier` shows whether either model learned anything. Many stock-prediction projects skip this, and it makes high accuracy look better than it is.
- **Avoids common mistakes.** Time-series data needs extra care (see [Methodology](#-methodology)).
- **Any outcome is a valid result.** RF wins, LR wins, a tie, or neither beats the baseline all say something real about the problem.

This sits between theory (Fama) and heavy deep-learning work (Fischer & Krauss, 2018). It asks whether simple, interpretable models are enough, or whether complexity actually helps.

---

## Data

| Item | Detail |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Ticker | NVDA (NVIDIA) |
| Period | Jan 2020 – Dec 2025 |
| Frequency | Daily |
| Raw fields | Open, High, Low, Close, Volume |

---

## Features and Target

There are four features, intentionally kept small so results stay easy to interpret:

| Feature | Definition |
|---|---|
| Daily Return | `Close_t / Close_{t-1} − 1` |
| Trading Volume | Shares traded that day |
| 5-Day Momentum | `Close_t / Close_{t-5} − 1` |
| 5-Day Volatility | Rolling std. dev. of daily returns (5 days) |

**Target (binary):**
- `1` → next day's close is **higher** than today's
- `0` → next day's close is **equal or lower**

Tomorrow's price is used *only* to build the label, never as an input.

---

## Models

| Model | Role | Setup |
|---|---|---|
| **Majority-class baseline** | Sanity check | `DummyClassifier` |
| **Logistic Regression** | Simple linear baseline | `StandardScaler` + `LogisticRegression` in a `Pipeline` |
| **Random Forest** | Flexible nonlinear model | `RandomForestClassifier` (100 trees, max depth 5, fixed seed) |

> **A note on wording:** Both models are machine learning algorithms. This project compares a *simple linear statistical baseline* against a *more flexible nonlinear model*. It is not "ML vs. statistics."

---

## Methodology

```
Download data → Build features + labels → Chronological 80/20 split → Train → Evaluate → Visualize
```

Choices made to keep the evaluation honest:

- **Chronological split.** The earliest 80% of days are for training and the latest 20% for testing. No random shuffling, so the model never trains on the future.
- **Boundary gap.** One day is dropped between train and test so the last training label doesn't depend on the first test-period price.
- **No scaler leakage.** `StandardScaler` is fit on training data only.
- **No dummy final label.** The last row is dropped because its "tomorrow" doesn't exist.
- **Identical inputs for every model.** Same features, same split, fixed random seed.

**Metric:** classification accuracy, plus confusion matrices to show *which* mistakes each model makes.

---

## Results

> *Experiments in progress. Table and chart will be added once complete.*

| Model | Test Accuracy |
|---|---|
| Majority-class baseline | _TBD_ |
| Logistic Regression | _TBD_ |
| Random Forest | _TBD_ |

![Model comparison](model_comparison.png)

**Key findings:** _TBD_

---

## Limitations

Stating these up front is part of doing the research properly.

- **One stock, one period.** Results may not generalize to other stocks or market regimes.
- **Small test set.** A gap of a few percentage points may not be statistically meaningful.
- **Accuracy is not profit.** Trading results depend on transaction costs, risk, and the size of returns, not just direction.
- **Only four features.** Chosen for interpretability. Richer features might change the outcome.

---

## Possible Extensions

- Statistical significance testing of the accuracy gap (e.g., McNemar's or a binomial test)
- Walk-forward (rolling window) validation
- Repeat across multiple stocks and sectors
- Add precision/recall, and compare against buy-and-hold returns

---

## Repository Structure

```
├── NVDA_Stock_Prediction.ipynb   # Main Colab notebook
├── NVDA_raw_data.csv             # Raw Yahoo Finance data
├── NVDA_processed_data.csv       # Features + target labels
├── model_comparison.png          # Accuracy comparison chart
└── README.md
```

## How to Run

1. Open `NVDA_Stock_Prediction.ipynb` in Google Colab.
2. Run all cells. Dependencies: `pandas`, `numpy`, `yfinance`, `scikit-learn`, `matplotlib`.

---

## References

1. Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* The Journal of Finance.
2. Fischer, T., & Krauss, C. (2018). *Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions.* European Journal of Operational Research.

---

*Educational research project. Not financial advice.*
