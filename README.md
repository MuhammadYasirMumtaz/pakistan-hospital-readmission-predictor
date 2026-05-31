# Pakistan Hospital Readmission Predictor

AI-powered 30-day hospital readmission risk assessment system built for clinical decision support.

## Live Demo
[Try it here](https://pakistan-hospital-readmission-predictor-production.up.railway.app)

## Project Overview
This system predicts whether a diabetic patient will be readmitted to hospital within 30 days using machine learning, providing instant risk assessment with AI explanations for clinical decision support.

## Model Performance
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.6835 |
| Accuracy | 73.58% |
| Recall | 48.39% |
| Training Records | 101,766 patients |

## Tech Stack
- **ML Model:** CatBoost Classifier
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Backend:** Flask (Python)
- **Database:** MySQL with ETL Pipeline
- **Frontend:** Bootstrap 5 (Mobile Responsive)
- **Deployment:** Docker + Railway
- **Dashboard:** Power BI

## Key Features
- ✅ Real-time 30-day readmission risk prediction
- ✅ Per-patient SHAP waterfall explanation
- ✅ PDF report download
- ✅ Live analytics dashboard
- ✅ Model performance comparison page
- ✅ MySQL prediction logging
- ✅ Mobile responsive UI
- ✅ Dockerized deployment

## Project Structure
\`\`\`
hospital_project/
├── app.py                  # Flask web application
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── models/                 # Trained ML models
├── notebooks/              # Jupyter analysis notebooks
│   ├── day1_eda.ipynb      # Data exploration & cleaning
│   ├── day2_database.ipynb # MySQL ETL pipeline
│   └── day3_model.ipynb    # ML model training
├── templates/              # HTML pages
│   ├── index.html          # Patient input form
│   ├── result.html         # Prediction results + SHAP
│   ├── analytics.html      # Live analytics dashboard
│   └── model.html          # Model performance page
├── static/                 # Charts and assets
└── data/                   # Dataset files
\`\`\`

## Why CatBoost over XGBoost?
65% of our features are categorical. CatBoost handles them natively without label encoding, achieving higher AUC (0.6835 vs 0.6795) and recall (48.39% vs 43.20%). Missing a high-risk patient is far more dangerous than a false alarm — so recall matters more than accuracy.

## 📈 Feature Engineering
Created 5 new predictive features:
- `total_visits` — sum of all hospital visits
- `inpatient_ratio` — proportion of serious admissions (became #2 most important feature!)
- `emergency_ratio` — proportion of emergency visits
- `medication_per_visit` — average medications per visit
- `lab_procedure_ratio` — lab tests per hospital day

## SHAP Explainability
Every prediction comes with a SHAP waterfall chart explaining exactly why the patient was flagged — which factors pushed the risk up (red) or down (blue). This makes the AI transparent and trustworthy for medical staff.

## Research Benchmark
Published research on this exact UCI Diabetes 130-US Hospitals dataset reports AUC of 0.58-0.72. Our 0.68 places us in the "Good Engineering" category, achieved through feature engineering, diagnosis code grouping and CatBoost optimization.

## Author
**Muhammad Yasir Mumtaz** — CS Student, SSUET Karachi

## Disclaimer
This system is for clinical decision support only. Always consult qualified medical professionals before making clinical decisions.
