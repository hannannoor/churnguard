# ChurnGuard

Streamlit app for the MLDP project (CAI2C08). Scores a telecom customer's churn risk live, explains
what drove that score, and ranks the retention actions that would reduce it.

**Live app:**  https://hannan-noor-repo-urnb4gmsdju7kjo5dz585d.streamlit.app/
**Model and analysis:** [MLDP-C25U02-Apr2026](https://github.com/alvin-ps-tan/MLDP-C25U02-Apr2026)
(branch `hannan-noor`)

---

## What it does

**Live scoring.** No submit button — the probability updates as soon as any input changes, and the
current score stays pinned to the corner while you scroll.

**Attribution.** Shows which features pushed this particular prediction up or down. Contribution is
the feature's scaled value multiplied by its fitted coefficient, read from the deployed model rather
than hard-coded, so the explanation always matches the model actually making the prediction.

**Retention levers.** The main feature. For each action the business could realistically take, the
app re-runs the model on the same customer with that one field changed and reports how far the risk
drops — for example *move to a 2-year contract: 42% → 13%*. Levers are filtered so only sensible
options appear: anything the customer already has is skipped, add-ons disappear when they have no
internet service, and actions that would raise risk are excluded.

Engineered features are recomputed for every scenario, so adding a service correctly updates the
services-held count rather than scoring a stale row.

These are **modelled associations, not causal effects.** The model learned that customers on two-year
contracts churn less; it cannot show that moving someone onto one would cause that drop.

## The model

Logistic Regression pipeline (`StandardScaler` + `LogisticRegression`, `C=2.0`), trained on 7,032
customers with 34 features. Cross-validated weighted F1 of 0.796, test weighted F1 of 0.787.

Full analysis, model selection and evaluation are in the notebook repository linked above.
`model.pkl` here is written directly by Section 7 of that notebook, so the two cannot drift apart.

## Running locally

```bash
pip install -r churn-guard/requirements.txt
streamlit run churn-guard/streamlit_app.py
```

## Layout

```
.
├── .streamlit/
│   └── config.toml           pinned dark theme, so it renders the same everywhere
└── churn-guard/
    ├── streamlit_app.py      the app
    ├── model.pkl             trained pipeline, written by the notebook
    └── requirements.txt      dependencies
```

`scikit-learn` is pinned to the version that created the pickle. Unpinning it is the usual cause of a
model that loads locally and fails on deploy.

## Deployment

Hosted on Streamlit Community Cloud with entrypoint `churn-guard/streamlit_app.py` on `main`. Pushing
to `main` redeploys automatically.

---

Decision-support tool built for coursework. It supports retention decisions rather than replacing
commercial judgement.
