# Telco Customer Churn Prediction

MLDP project (CAI2C08) — predicting which telecom customers are likely to **churn** (cancel their
service), so the business can step in with retention offers *before* they leave. Keeping an existing
customer is far cheaper than winning a new one, so catching churners early directly protects revenue.

**Live app:** https://hannan-noor-repo-urnb4gmsdju7kjo5dz585d.streamlit.app/
**App source:** [hannannoor/hannan-noor-repo](https://github.com/hannannoor/hannan-noor-repo)

---

## Dataset

Telco Customer Churn (IBM sample) — 7,043 customers, 21 columns. Target is `Churn` (Yes/No) at a
**26.5%** churn rate, so the classes are moderately imbalanced.

Eleven rows had a blank `TotalCharges`. All eleven were customers with `tenure = 0` who had not been
billed yet, so the value was genuinely absent rather than missing at random, and they were dropped.
That leaves **7,032** customers.

## Approach

The notebook follows CRISP-DM: business understanding, data understanding, data preparation,
modelling, tuning, evaluation, and deployment.

**Feature engineering.** Three features were created and two kept:

| Feature | What it captures | Kept |
|---|---|---|
| `tenure_group` | Churn risk is sharply non-linear in tenure — ~48% for new customers, under 10% for veterans | Yes |
| `num_services` | How embedded a customer is, counting active services | Yes |
| `avg_monthly_charges` | Intended as spending intensity | **No** |

`avg_monthly_charges` was dropped during feature selection. It correlated **0.996** with
`MonthlyCharges`, sat $1.75 away from it on average, and fell within $5 for ~94% of customers — a
duplicate rather than new signal. Because Logistic Regression splits its coefficient between
correlated predictors, keeping it weakened and destabilised the coefficient ranking.

After one-hot encoding with `drop_first=True`, the model uses **34 features**.

**Metric choice.** Weighted F1 leads, because accuracy is misleading here — always predicting "No"
scores 73.4% while catching zero churners. Recall on the churn class is tracked alongside it, since
the business goal is catching churners before they leave.

## Results

All models were tuned with `RandomizedSearchCV` (5-fold CV, no more than 3 values per hyperparameter).

| Model | CV weighted F1 | Test weighted F1 | Churn recall | Accuracy |
|---|---|---|---|---|
| Baseline (Dummy) | — | 0.622 | 0.000 | 0.734 |
| Decision Tree | 0.782 | 0.779 | 0.596 | 0.778 |
| Random Forest | 0.794 | 0.781 | 0.495 | 0.790 |
| Gradient Boosting | 0.793 | 0.783 | 0.495 | 0.792 |
| **Logistic Regression** | **0.796** | **0.787** | 0.529 | 0.793 |

**Final model: Logistic Regression** (`C=2.0`, `class_weight=None`), inside a pipeline with
`StandardScaler`.

It wins narrowly on the numbers, but the stronger case is practical: its coefficients name the exact
churn drivers a retention team can act on, it retrains in seconds, and it produces a small model that
deploys easily. The tree ensembles offered no accuracy gain to justify being heavier and less
explainable.

**Reading the recall.** Churn recall of 0.529 at precision 0.63 means a flagged customer is roughly
**2.4× more likely** to churn than a randomly chosen one, against the 26.5% base rate. The baseline
catches 0%. Recall is held down partly by tuning on weighted F1, which favours the 73.5% majority
class, and partly by the default 0.5 threshold — which can be lowered to trade precision for recall
without retraining.

**What drives churn.** By churn-rate spread across categories: `Contract` 39.9%, `InternetService`
34.5%, `OnlineSecurity` 34.4%, `TechSupport` 34.2%, `PaymentMethod` 30.0%. At the other end, `gender`
separates churn by 0.8% and `PhoneService` by 1.8%.

## Repository

```
MLDP-C25U02-Apr2026/
├── hannan-noor/
│   ├── MLDP_notebook.ipynb    full analysis and modelling
│   ├── churn.csv              dataset
│   └── model.pkl              trained pipeline
└── CAI2C08 ... Project Specification.pdf
```

The Streamlit app lives in a [separate repository](https://github.com/hannannoor/hannan-noor-repo)
so it can be deployed on its own. Section 7 of the notebook saves `model.pkl` to both locations, so
the deployed app cannot drift out of sync with the notebook that trained it.

## Running the notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook hannan-noor/MLDP_notebook.ipynb
```

Run all cells in order. The final cell writes `model.pkl`.

## Limitations

- Roughly half of real churners are missed at the default 0.5 threshold.
- Everything here is association, not causation.
- The dataset is a single snapshot with no time dimension, so behavioural changes that often precede
  churn are invisible to the model.
- Trained on one provider's customers; it should not be assumed to transfer elsewhere without
  retraining.
