# Pulse of Prevention: Analyzing Heart Health for Better Outcomes

A data analytics case study for HealthPulse Analytics on the Kaggle Heart Disease dataset (`heart.csv`): who has heart disease in this data, which factors go with it, and what preventive steps follow. All 25 case-study questions are answered in a Python notebook (Pandas, Matplotlib, Seaborn, SciPy, scikit-learn); every number below comes from that notebook.

---

## Read this first: two data issues

1. **Duplicates.** `heart.csv` has 1,025 rows but only **302 are unique**; 723 are exact repeats. Every answer is therefore shown twice: **All rows (1,025)**, as the brief's hints compute it, and **Unique rows (302)**, which avoids counting a patient several times. Use the unique rows for statistical tests and models.
2. **Label check.** `target = 1` means heart disease in the brief. But patients with `target = 1` have *less* exercise angina (14% vs 55%), *lower* ST depression (0.59 vs 1.59), *fewer* affected vessels (average 0.34 vs 1.17) and a *higher* max heart rate (158 vs 139) than `target = 0` (unique rows). Clinically these signs go with disease, so the label may be reversed in this Kaggle copy. This repository follows the brief's definition and reports results as they are in the data; directions of effect should be checked against the dataset's source before clinical conclusions are drawn.

---

## Executive Summary

- **Patients:** 1,025 rows (302 unique). Average age 54.4, 68% to 70% male, average resting BP 131.6 mm Hg, average cholesterol 246 mg/dl.
- **target = 1:** 51.3% of all rows (526), 54.3% of unique rows (164).
- **Strongest associations with target** (correlation, unique rows): exercise angina -0.44, chest pain type +0.43, ST depression (oldpeak) -0.43, max heart rate +0.42, major vessels -0.41. Cholesterol (-0.08) and fasting blood sugar (-0.03) are weak.
- **Model:** logistic regression on all 13 features reaches a cross-validated AUC of 0.905 (accuracy 83%) on unique rows; on all rows it looks better (AUC 0.92) only because duplicates leak between folds.
- **Not possible:** Advanced Q5 (survival rates): the data has no follow-up or survival information.

---

## Dataset and Cleaning

- `heart.csv`: 1,025 rows x 14 columns (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target).
- **Missing values:** none. **Data types:** all numeric; coded columns (sex, cp, fbs, restecg, exang, slope, ca, thal, target) are categorical codes.
- **Duplicates:** 723 exact duplicate rows; both versions analysed.
- **Out-of-range codes:** `ca = 4` (18 rows, 4 unique) and `thal = 0` (7 rows, 2 unique) are outside the data dictionary; they are few, so they are kept and flagged in `data/heart_cleaned.csv`.
- **Outliers (1.5 x IQR, unique rows):** resting BP 9, cholesterol 5 (max 564), max heart rate 1, oldpeak 5, age 0. They are plausible clinical values, so they are kept.
- **Normalization:** min-max scaled columns (`*_norm`) are saved in `heart_cleaned.csv`; the logistic model uses standardisation so coefficients are comparable.
- `cp` codes 0 to 3 are not labelled in the brief, so they are reported as codes.

---

## Answers to the Case-Study Questions

### Basic

| Question | All rows (1,025) | Unique rows (302) |
| :--- | :--- | :--- |
| Q1 Average age | 54.43 | 54.42 |
| Q2 Gender distribution | Male 713 (69.6%), female 312 (30.4%) | Male 206 (68.2%), female 96 (31.8%) |
| Q3 Average resting BP | 131.6 mm Hg | 131.6 mm Hg |
| Q4 Fasting sugar > 120 | 153 patients | 45 patients |
| Q5 Chest pain types | 0, 1, 2, 3 (counts 497, 167, 284, 77) | 0, 1, 2, 3 (counts 143, 50, 86, 23) |
| Q6 Max heart rate achieved | 202 | 202 |
| Q7 Exercise-induced angina | 33.7% | 32.8% |
| Q8 Average cholesterol | 246.0 mg/dl | 246.5 mg/dl |
| Q9 Resting ECG = 2 | 15 patients | 4 patients |
| Q10 Major vessels (ca) | 0: 578, 1: 226, 2: 134, 3: 69, 4: 18 | 0: 175, 1: 65, 2: 38, 3: 20, 4: 4 |

### Medium

| Question | All rows (1,025) | Unique rows (302) |
| :--- | :--- | :--- |
| M1 Age vs cholesterol | r = 0.220 (p < 0.001) | r = 0.207 (p < 0.001) |
| M2 Chest pain by age group | Type 0 rises with age (35% under 45 to 59% at 55-64); types 1 and 2 are more common under 55 | Same pattern: type 0 34.5% (<45), 40.9%, 58.5% (55-64), 46.3% (65+) |
| M3 Max heart rate by exercise angina | No angina 155.3, angina 136.8 (p < 0.001) | 155.6 vs 137.2 (p < 0.001) |
| M4 Resting BP, male vs female | Male 130.7, female 133.7; p = 0.018 (significant) | Male 130.9, female 133.1; p = 0.34 (not significant) |
| M5 Fasting sugar vs target | target = 1 in 52.2% (sugar normal) vs 46.4% (high); chi-square p = 0.22 | 54.9% vs 51.1%; p = 0.76. No significant link |
| M6 Major vessels vs target | target = 1: ca 0 70.8%, ca 1 29.2%, ca 2 15.7%, ca 3 13.0% (ca 4: 83%, n = 18) | ca 0 74.3%, ca 1 32.3%, ca 2 18.4%, ca 3 15.0% (ca 4: n = 4) |
| M7 Average oldpeak by chest pain | cp 0 1.44, cp 1 0.32, cp 2 0.78, cp 3 1.38 | 1.38, 0.32, 0.81, 1.39 |
| M8 Thalassemia (thal) vs target | target = 1: thal 2 75.7%, thal 3 22.0%, thal 1 32.8% | thal 2 78.2%, thal 3 23.9%, thal 1 33.3% |
| M9 Common risk-factor combinations (target = 1) | cp 2, fbs 0, exang 0, thal 2: 133 (25.3%); cp 1 same: 97; cp 0 same: 73 | Same top three: 41 (25.0%), 30, 23 |
| M10 Measurements, target 1 vs 0 | Max HR 158.6 vs 139.1; oldpeak 0.57 vs 1.60; age 52.4 vs 56.6; BP 129.3 vs 134.1; chol 241.0 vs 251.3 | Max HR 158.4 vs 139.1; oldpeak 0.59 vs 1.59; age 52.6 vs 56.6; BP 129.3 vs 134.4; chol 242.6 vs 251.1 (p = 0.16, not significant) |

### Advanced

| Question | Answer |
| :--- | :--- |
| A1 Combined age, cholesterol, BP | Share with target = 1 by number of risk flags (age 55+, chol 240+, BP 140+), unique rows: 0 flags 73.8% (n=65), 1 flag 60.2%, 2 flags 41.5%, 3 flags 37.5% (n=40). More flags goes with a lower share, the opposite of the usual expectation (see label check). In a 3-variable logistic model, age is significant (odds ratio 0.96 per year, p = 0.002); cholesterol (p = 0.58) and BP (p = 0.14) are not; AUC 0.65 |
| A2 Strongest correlation with target | Unique rows: exercise angina -0.44, chest pain type +0.43, oldpeak -0.43, max heart rate +0.42. All rows: oldpeak -0.438, exang -0.438, cp +0.435, thalach +0.423. Among continuous measurements: oldpeak and max heart rate |
| A3 Logistic regression, all features | Unique rows: 5-fold CV AUC 0.905, accuracy 83.1%; hold-out (25%) accuracy 80.3%, AUC 0.875 (confusion matrix: 25 and 36 correct, 10 and 5 wrong). All rows score higher (AUC 0.92, accuracy 84.6%) only because duplicate rows leak across folds. Largest standardised effects: chest pain type +0.83, major vessels -0.79, sex -0.77, oldpeak -0.58, thal -0.52, max heart rate +0.50, exercise angina -0.45 |
| A4 ST slope vs chest pain type | Unique rows: slope 2 is most common for cp 1 (72%) and cp 2 (56%); slope 1 dominates cp 0 (59%) and cp 3 (48%). Slope 0 is rare (4% to 13%) |
| A5 Survival by thalassemia type | **Not possible:** one row per patient, with no follow-up time or outcome date. Descriptive substitute: share with target = 1 by thal and age group is in the notebook |

---

## Recommendations (read with the label check)

- **Prioritise screening** for patients with exercise-induced angina, abnormal ST depression and multiple affected vessels: they are the strongest discriminators in the data.
- **Do not rely on cholesterol or fasting sugar alone;** both are weak in this dataset (correlations -0.08 and -0.03; fasting sugar not significant).
- **Build the risk score on unique patients** and validate it on a separate hospital sample: more than two thirds of this file is repeated rows.
- **Confirm the target coding with the data source** before using any direction of effect for patient guidance.
- **Further research:** collect follow-up outcomes so survival and long-term risk can be analysed.

---

## Repository Structure

```text
├── data/
│   ├── heart.csv
│   └── heart_cleaned.csv
├── notebooks/
│   └── Pulse_of_Prevention_Heart_Health_Analysis.ipynb
├── screenshots/
│   └── heart_health_dashboard.png
├── presentation/
│   └── Pulse of Prevention Heart Health Analysis.pptx
└── README.md
```

---

## Limitations

- **Association, not causation.** One hospital snapshot, ages 29 to 77.
- **Only 302 distinct records,** so subgroup results (for example ca = 4, thal = 0, restecg = 2) rest on very few patients.
- **No survival or follow-up data,** and the `target` label direction needs checking.
- `cp`, `slope` and `thal` codes are not labelled in the brief.
