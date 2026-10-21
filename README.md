# ChurnGuard

Telecom churn analysis and an interactive retention decision-support app.

**Built with:** Python · scikit-learn · Streamlit

**Subject results:** MLDP: Distinction  
*Subject grades; individual project marks were not supplied.*

## What it includes

- CRISP-DM notebook with the telco dataset, model comparison and feature engineering.
- A Streamlit app with customer-level scoring, coefficient attribution and retention scenarios.
- Matching analysis and deployed model artifacts.

## Project status

Source and model artifacts checked; inference/UI run still pending. Retention scenarios show modelled associations, not causal treatment effects. The pitch has unfinished header placeholders and should be reviewed before it is used publicly.

## Folder guide

| Folder | Contents |
| --- | --- |
| [`src/`](src/) | Original implementation, assets and project configuration |
| [`docs/`](docs/) | Available specifications, proposals, reports, diagrams and project context |

## Run or open

1. Open `src/Analysis/MLDP_notebook.ipynb` from that folder so `churn.csv` resolves.
2. Use an environment compatible with the app requirements; the inner file specifies scikit-learn 1.8.0 and joblib 1.4.2.
3. From this project folder:

```text
cd src/Application
python -m pip install -r churn-guard/requirements.txt
python -m streamlit run churn-guard/streamlit_app.py
```

After retraining, copy the new model into `src/Application/churn-guard/`. The notebook's optional export path refers to the old checkout.

## Supporting documents

| File | Format |
| --- | --- |
| [project-context.md](docs/project-context.md) | MD |
| [mldp-churnguard-pitch.pptx](docs/reports-and-presentations/mldp-churnguard-pitch.pptx) | PPTX |
| [mldp-mldp-program-codes-submission-declaration.docx](docs/reports-and-presentations/mldp-mldp-program-codes-submission-declaration.docx) | DOCX |
| [mldp-program-codes-submission.docx](docs/reports-and-presentations/mldp-program-codes-submission.docx) | DOCX |
| [mldp-project-specification.pdf](docs/specifications/mldp-project-specification.pdf) | PDF |

## Notes

API keys and demonstration passwords detected in the portfolio copies use
placeholders. Configure your own services before running integrations. Original
source credits and bundled third-party licences remain with their files.
