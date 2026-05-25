# Predicting the Economic Value of Healthcare Automation
### A causal-inference and Monte Carlo approach to no-show management for private clinics

This repository hosts the analytical backbone of my Bachelor's thesis (TFG) at Universidad Pontificia Comillas — a quantitative case study estimating how much economic value an automated SMS-reminder strategy can recover from missed medical appointments, and translating that estimate into a defensible value-based pricing recommendation for a Spanish private clinic.

The project is also the empirical foundation for **aiXtensa**, a healthcare-automation venture I am developing in parallel.

---

## The problem

No-shows are the silent margin killer of outpatient care. Every empty slot is unrecoverable revenue, idle clinical staff time, and a patient who needed the appointment slot. The literature reports no-show rates of 15–30% in public systems and 8–15% in private clinics. The intuitive question — *"how much would it cost a clinic to do nothing about this?"* — is almost never answered rigorously because two things are usually conflated:

1. **Predicting** who will not show up (a supervised learning problem).
2. **Estimating** how much a reminder would actually change that outcome (a *causal* problem — observational data, confounded by selection).

This project keeps the two problems separate by design, then composes the answers into a Monte Carlo financial simulation that propagates uncertainty from the causal estimate all the way through to a recommended price point.

---

## Methodology at a glance

The stack is intentionally minimal and defensible end-to-end. Each block feeds the next.

| Layer | Method | What it produces | Why |
|---|---|---|---|
| **Risk model** | XGBoost + isotonic / Platt calibration | Per-patient no-show probability | Discrimination needs gradient boosting; downstream Monte Carlo needs *calibrated* probabilities, not just ranking |
| **Explainability** | SHAP (TreeExplainer, full test set) | Global + local feature attribution | Defensibility under oral examination; sanity-check signs against domain |
| **Causal layer** | Stabilised IPW with cluster bootstrap by patient | ATE of SMS receipt on attendance (pp), with CI | Treatment was non-randomly assigned (selection bias visible in EDA); IPW recovers an interpretable absolute effect under no-unobserved-confounders |
| **Segmentation** | K-means on calibrated risk + behavioural features (GMM fallback) | Patient archetypes with differentiated baseline risk | Lets us price the service against the segments that actually move the economic needle |
| **Pricing simulation** | Monte Carlo (10,000 iterations) on cost-recovery distribution | Distribution of recoverable € per appointment per month | Translates a noisy causal estimate into a *probabilistic* pricing recommendation rather than a single point estimate |
| **Market parameters** | Light webscraping of Spanish private-clinic price lists | Empirical priors for fee, capacity, and cost parameters | Anchors the simulation to the Spanish market rather than the source dataset's geography |

The **Monte Carlo pricing model** is the headline contribution: it is what turns a methodologically correct ATE into a number a clinic owner can actually act on. The recommendation is expressed as fixed-fee + variable (share of recovered value) tiers at the P10 / P50 / P90 of the simulated distribution, so the buyer can choose their own risk posture rather than receiving a single point quote.

---

## Methodological discipline (what this project deliberately does *not* do)

These choices are part of the contribution — for a thesis project, knowing what to leave out matters as much as what to add.

- **`SMS_received` is excluded from the XGBoost risk model.** It is the treatment variable; including it would leak the causal signal we estimate separately.
- **No outcome-based target encoding.** `Neighbourhood` is frequency-encoded to avoid leaking `Showed_up` into either the risk or the propensity model.
- **Time-based train/test split**, not random — the dataset contains repeat patients, and a random split would inflate apparent performance.
- **Patient-history features anchored to *scheduling* time**, not to observation order, so historical statistics never include information that would have been unknown at the moment of the intervention decision.
- **Single estimator (IPW), not DoubleML / AIPW / causal forests.** Marginal robustness gain is not worth the defence-exposure cost of higher-machinery estimators; they are acknowledged in the limitations as future work.
- **Causal language used consistently**: the IPW estimate is reported as an association under adjustment, never as proof of causation, in line with the observational nature of the data.
- **Left-censoring is declared.** Patient-history features describe only *observed* history within the ~6-week data window, not the true patient–clinic relationship. The thesis text addresses this explicitly rather than papering over it.

---

## Dataset

**Source:** [Medical Appointment No-Shows — Kaggle (Joni Arroba)](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
**Coverage:** ~107,000 outpatient appointments, Vitória (Espírito Santo), Brazil, 2016
**Variables:** 15 columns — demographics, comorbidities (hypertension, diabetes, alcoholism), socio-economic proxies (Bolsa Família scholarship), neighbourhood, scheduling lead time, SMS reminder flag, attendance outcome.

The raw CSV is **not committed** (see `.gitignore`). To reproduce locally:

```bash
# After cloning, place the Kaggle CSV here:
datos/brutos/healthcare_noshows.csv
```

The dataset is used as a **behavioural proxy**. Economic parameters are recalibrated to the Spanish private-clinic market via INE statistics and the webscraped fee priors in Notebook 07. The geographic mismatch is declared explicitly in the limitations section of the thesis rather than hidden.

---

## Repository structure

```
.
├── notebooks/
│   ├── 01_limpieza_y_variables.ipynb          # Data cleaning + feature engineering + time split
│   ├── 02_exploracion_datos.ipynb             # EDA + selection-bias diagnosis (motivates IPW)
│   ├── 03_modelo_xgboost_calibracion_shap.ipynb  # Risk model + calibration + SHAP
│   ├── 04_inferencia_causal_ipw.ipynb         # Stabilised IPW + cluster bootstrap [WIP]
│   ├── 05_segmentacion_pacientes.ipynb        # K-means / GMM patient segments [WIP]
│   ├── 06_pricing_montecarlo.ipynb            # Monte Carlo pricing simulation [WIP]
│   └── 07_webscraping_mercado.ipynb           # Spanish market parameters [WIP]
├── src/
│   ├── preparacion.py    # Cleaning + feature engineering (used by NB01)
│   ├── eda.py            # Plotting and diagnostic helpers (used by NB02)
│   ├── modelado.py       # Training, calibration, SHAP utilities (used by NB03)
│   └── rutas.py          # Project paths
├── app/
│   ├── calculadora_roi.py   # Streamlit ROI calculator (prototype)
│   └── dashboard.py         # Streamlit dashboard (prototype)
├── datos/
│   ├── brutos/           # Raw Kaggle CSV (gitignored — fetch manually)
│   └── procesados/       # Train/test/full clean splits (regenerable from NB01)
├── outputs/
│   ├── figuras/          # Plots (gitignored — regenerable from notebooks)
│   ├── modelos/          # Pickled trained models (gitignored)
│   ├── reportes/         # CSV / JSON tabular exports (gitignored)
│   └── bootstrap/        # Bootstrapped ATE distribution — bridge from NB04 → NB06
├── requirements.txt
├── requirements-lock.txt
├── LICENSE
└── README.md
```

Notebook narrative is in **Spanish** (this is a thesis deposited in Spanish at a Spanish university). Code identifiers are bilingual — Python APIs in English, domain-level helpers in Spanish for coherence with the written document. The README and `requirements.txt` are in English.

---

## Reproduction

```bash
# 1. Clone
git clone <repo-url> aixtensa-noshow-tfg
cd aixtensa-noshow-tfg

# 2. Fetch the dataset from Kaggle and place it at:
#    datos/brutos/healthcare_noshows.csv

# 3. Create the environment (Python 3.14)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
#  …or, to reproduce the exact transitive pin set:
#  pip install -r requirements-lock.txt

# 4. Run the notebooks in order
jupyter notebook notebooks/
```

All stochastic processes (XGBoost training, K-means initialisation, bootstrap, Monte Carlo) are seeded. Intermediate artefacts are versioned (e.g. `train_v1.csv`, `xgboost_v1.json`, `ate_bootstrap_v1.npy`) so results are stable across reruns.

---

## Project status

| Stage | Notebook | Status |
|---|---|---|
| Data cleaning + feature engineering | `01_limpieza_y_variables` | Complete |
| Exploratory analysis + selection-bias diagnosis | `02_exploracion_datos` | Complete |
| Calibrated XGBoost + SHAP | `03_modelo_xgboost_calibracion_shap` | Complete |
| Stabilised IPW with cluster bootstrap | `04_inferencia_causal_ipw` | In progress |
| Patient segmentation | `05_segmentacion_pacientes` | In progress |
| Monte Carlo pricing simulation | `06_pricing_montecarlo` | In progress |
| Spanish market parameter scraping | `07_webscraping_mercado` | In progress |
| Streamlit ROI calculator + dashboard | `app/` | Prototype |
| Written thesis document (Spanish) | `memoria/` | Pending — deposit June 2026 |

Updates land on `main` as each notebook is finalised.

---

## Selected references

- Rosenbaum, P. R., & Rubin, D. B. (1983). *The central role of the propensity score in observational studies for causal effects*. Biometrika, 70(1).
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge UP.
- VanderWeele, T. J., & Ding, P. (2017). *Sensitivity Analysis in Observational Research: Introducing the E-Value*. Annals of Internal Medicine, 167(4).
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NIPS '17.

---

## Author

**Sergio Cuervo Arango** — Final-year student, *Administración y Dirección de Empresas + Business Analytics* (E-2 + BA), Universidad Pontificia Comillas (ICADE).
Founder, [aiXtensa](#) — voice and workflow automation for Spanish private clinics.

Built in parallel with my ADE thesis, which develops the business model around the analytical results produced here.

---

## License

MIT — see [`LICENSE`](LICENSE).

The dataset is not redistributed; please obtain it directly from Kaggle under its own licence.
