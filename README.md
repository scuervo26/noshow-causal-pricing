# Predicting the Economic Value of Healthcare Automation
### A causal-inference and Monte Carlo approach to no-show management for private clinics

Bachelor's thesis (TFG) at Universidad Pontificia Comillas. The project estimates how much economic value an automated SMS-reminder strategy can recover from missed medical appointments, and translates that estimate into a value-based pricing recommendation for a Spanish private clinic.

It is also the empirical foundation for **aiXtensa**, a healthcare-automation venture I am building in parallel.

---

## The problem

Outpatient no-shows quietly destroy clinic margin: empty slots are unrecoverable revenue, idle clinical time, and unmet patient demand. Two questions usually get conflated:

1. **Who** is likely to miss their appointment? (a supervised learning problem)
2. **How much** would a reminder actually change that outcome? (a *causal* problem — observational data, confounded by selection)

This project keeps them separate by design, then composes the answers in a Monte Carlo financial simulation that propagates uncertainty from the causal estimate all the way to a recommended price tier.

---

## Methodology

| Layer | Method | What it produces |
|---|---|---|
| Risk model | Calibrated XGBoost (isotonic / Platt) | Per-patient no-show probability |
| Explainability | SHAP (TreeExplainer, full test set) | Global + local feature attribution |
| Causal layer | Stabilised IPW + cluster bootstrap by patient | ATE of SMS receipt on attendance (pp), with CI |
| Segmentation | K-means on calibrated risk + behavioural features | Patient archetypes with differentiated baseline risk |
| Pricing simulation | Monte Carlo, 10,000 iterations | Distribution of recoverable € per appointment per month |
| Market parameters | Light webscraping of Spanish private-clinic fees | Empirical priors anchored to the Spanish market |

The **Monte Carlo pricing model** is the headline contribution: it turns a methodologically clean ATE into something a clinic owner can act on. The recommendation is delivered as fixed-fee + variable (share of recovered value) tiers at the P10 / P50 / P90 of the simulated distribution, so the buyer chooses their own risk posture instead of receiving a single point quote.

Key methodological choices: `SMS_received` is excluded from the risk model (it is the treatment variable); `Neighbourhood` is frequency-encoded (no outcome leakage); the train/test split is time-based because patients repeat; the IPW estimate is reported as an *association under adjustment*, not as proof of causation.

---

## Dataset

[Medical Appointment No-Shows — Kaggle (Joni Arroba)](https://www.kaggle.com/datasets/joniarroba/noshowappointments) — ~107,000 outpatient appointments, Vitória (ES, Brazil), 2016. Used as a behavioural proxy; economic parameters are recalibrated to Spain via INE and the webscraped fee priors in Notebook 07.

The raw CSV is not committed. Place it at `datos/brutos/healthcare_noshows.csv` after cloning.

---

## Repository structure

```
notebooks/   01 → 07: cleaning, EDA, XGBoost+SHAP, IPW, segmentation, Monte Carlo, scraping
src/         preparacion.py, eda.py, modelado.py, rutas.py — logic backing the notebooks
app/         Streamlit prototypes (ROI calculator, dashboard)
datos/       brutos/ (raw — gitignored), procesados/ (regenerable from NB01)
outputs/     figuras/, modelos/, reportes/, bootstrap/ (regenerable from notebooks)
```

Notebook narrative is in Spanish (deposited at a Spanish university); README and code APIs stay in English.

---

## Reproduction

```bash
git clone https://github.com/scuervo26/noshow-causal-pricing.git
cd noshow-causal-pricing

# Place the Kaggle CSV at: datos/brutos/healthcare_noshows.csv

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or requirements-lock.txt for exact pins

jupyter notebook notebooks/
```

All stochastic processes (XGBoost, K-means, bootstrap, Monte Carlo) are seeded; intermediate artefacts are versioned (`train_v1.csv`, `xgboost_v1.json`, `ate_bootstrap_v1.npy`) for stable reruns.

---

## Project status

| Notebook | Stage | Status |
|---|---|---|
| `01_limpieza_y_variables` | Data cleaning + feature engineering + time split | Complete |
| `02_exploracion_datos` | EDA + selection-bias diagnosis | Complete |
| `03_modelo_xgboost_calibracion_shap` | Calibrated XGBoost + SHAP | Complete |
| `04_inferencia_causal_ipw` | Stabilised IPW + cluster bootstrap | In progress |
| `05_segmentacion_pacientes` | K-means / GMM patient segments | In progress |
| `06_pricing_montecarlo` | Monte Carlo pricing simulation | In progress |
| `07_webscraping_mercado` | Spanish market parameter scraping | In progress |
| `app/` | Streamlit ROI calculator + dashboard | Prototype |

Updates land on `main` as each notebook is finalised.

---

## Author

**Sergio Cuervo Arango** — *Administración y Dirección de Empresas + Business Analytics* (E-2 + BA), Universidad Pontificia Comillas (ICADE). Founder, aiXtensa.

MIT licensed (see [`LICENSE`](LICENSE)). Dataset is not redistributed; obtain it from Kaggle under its own licence.
