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

## Why These Two Models?

### Logistic Regression: the simple baseline

**What it is:** A model that takes a weighted sum of the features (return, volume, momentum, volatility) and turns it into a probability that tomorrow is an "up" day. If the probability is above 50%, it predicts UP.

**How it thinks:** Each feature gets one weight, and the effects add up in a straight line. For example, "higher momentum nudges the probability up by this much."

| Strengths | Weaknesses |
|---|---|
| Fast and easy to train | Can only learn **linear** patterns |
| Weights are easy to interpret | Can't capture feature interactions on its own |
| Hard to overfit with only 4 features | May miss more complex behavior |

**Why I chose it:** It is the standard "start simple" model for binary classification. If a straight-line model can't find a signal, that tells us something. It also gives Random Forest a fair, interpretable opponent.

### Random Forest: the flexible challenger

**What it is:** A collection of many decision trees (100 here). Each tree asks a series of yes/no questions about the features, such as "Is volatility above X? Is momentum below Y?", and makes a prediction. The forest takes a **majority vote** across all trees.

**How it thinks:** Each tree trains on a random sample of the data and features, so the trees make different mistakes. Averaging them cancels out much of the noise.

| Strengths | Weaknesses |
|---|---|
| Captures **nonlinear** patterns and interactions | Harder to interpret than logistic regression |
| Needs no feature scaling | Can overfit noisy data, which is a real risk in finance |
| Works well on small tabular data | Doesn't extrapolate beyond training-data patterns |

**Why I chose it:** It is the natural next step up in complexity. Markets might behave differently depending on conditions, for instance "momentum matters only when volatility is low." A linear model can't express that, but a forest can. I limited it to `max_depth=5` to keep it from memorizing noise.

### Why compare them?

Both models see the **same four features and the same split**, so the only difference is how flexible the model is. This setup lets me ask: *does extra flexibility help, or does it just fit noise?*

| | Logistic Regression | Random Forest |
|---|---|---|
| Pattern type | Linear | Nonlinear |
| Interpretability | High | Medium |
| Overfitting risk | Low | Higher |
| Role in this project | Simple baseline | Flexible challenger |

---

## Why NVIDIA (NVDA)?

NVIDIA is a good stress test for prediction models because its price is driven by **several large, changing forces** at once:

- **The 2020 boom.** In 2020, demand for GPUs surged (gaming, data centers) and tech stocks rallied broadly. The start of the dataset captures a sharp market regime change.
- **The AI wave.** NVIDIA's chips became central to training and running AI models, which made the stock a bellwether for AI sentiment.
- **Semiconductor cycles.** Chip stocks swing with supply, demand, and industry-wide cycles, so NVDA is exposed to sector moves as well as company news.
- **Constant news flow.** Earnings, product launches, and macro events produce frequent large moves, so the data is noisy and has real volatility variation.
- **Accessible data.** It is heavily traded with clean, free daily history on Yahoo Finance.

**Why this matters for the research:** The 2020–2025 window includes several distinct market conditions: a pandemic-era rally, a downturn, and the AI surge. If a model finds a real pattern, a volatile stock like this gives it room to show up. If no pattern shows up even here, that is a meaningful result for the Efficient Market Hypothesis.

**Caveat:** NVDA rose strongly over this period, so "up" days may be more common than "down" days. That is why the majority-class baseline is essential. A model that always guesses UP could look good without learning anything.

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
