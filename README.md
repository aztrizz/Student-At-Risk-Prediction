# Student At-Risk Prediction

A binary classifier that predicts whether a student is at risk of failing a university module, using only data available up to and including **week 6** of term. The goal is an early-warning signal — flagging struggling students while there's still time to intervene, rather than predicting a failing grade after the fact.

This is a personal / educational project built to practice the full lifecycle of a tabular ML problem: multi-source data joining, feature engineering under a strict temporal constraint, leakage prevention, imbalanced classification, and honest model evaluation.

## Problem

Given a student's engagement and performance data through week 6 of a module, predict whether their final module score will fall below the pass mark (40).

**Constraint:** every feature must be genuinely knowable by week 6. This ruled out using anything from weeks 7+ or the final score itself, and required verifying — not assuming — that every joined data source (demographics, prior years, survey, module metadata) was actually available before or during that window.

## Data

~20,000 student-module records, joined from five sources:

| Source | Contents |
|---|---|
| `model_dataset_sample20k.csv` | Weekly attendance & VLE engagement (weeks 1–6), midterm score, final score |
| `dim_students.csv` | Age, education level, socio-economic rank, disabilities, first-gen status |
| `fact_enrolment_survey.csv` | Start-of-year survey: career clarity/confidence, belonging, self-efficacy, support satisfaction |
| `fact_progression.csv` | Prior-year academic outcomes (avg mark, modules passed) |
| `dim_modules.csv` | Assessment type, module year |

**Target:** `at_risk = 1` if `final_score < 40`, else `0`. Class balance is ~11.7% at-risk / 88.3% passing — a meaningfully imbalanced problem.

## Feature Engineering

- Converted raw weekly attendance counts into per-week attendance **rates**, then aggregated into early (weeks 1–3) vs. late (weeks 4–6) averages plus a trend (late − early), to capture trajectory rather than a single snapshot.
- Applied the same early/late/trend pattern to VLE metrics (logins, resource views, forum posts).
- Pulled each student's **prior-year** average mark via a `groupby().shift(1)` on sorted academic years — deliberately excluding the current year to avoid leaking the outcome back in as a feature. Added a `has_prior_year` flag to preserve the fact a value was imputed (53% of students had no prior year on record).
- Fixed a multi-label encoding issue: `disabilities` was stored as comma-separated strings (e.g. `"adhd,dyslexia"`). Naive one-hot encoding treated each unique *combination* as its own category, exploding into ~160 near-duplicate columns. Replaced with proper multi-label binarization — one clean binary flag per individual condition — reducing the feature space from 251 to 95 columns.
- Used one-hot encoding (not integer/label encoding) for `education` and `assessment_type`, since label-encoding implies a false ordinal relationship between unrelated categories.
- Missing survey responses and prior-year marks were median-imputed, with missingness flags preserved separately so the model can distinguish a real value from a filled-in placeholder.

## Data Leakage Safeguards

- `final_score` is touched in exactly one line (to construct the label) and is explicitly excluded from the feature set.
- Prior-year data uses `.shift(1)`, never the current year.
- Enrolment survey and demographics were confirmed to be genuinely known before week 6 (start-of-year survey, static student facts), not just topically related.
- An explicit `non_feature_cols` exclusion list acts as a final gate at training time, independent of upstream feature-building logic.
- **Known gap:** cross-validation currently uses `StratifiedKFold`, which does not guarantee a given student's records stay within a single fold if they appear in multiple modules/years. `GroupKFold` (grouped by `student_id`) would close this identity-leakage risk — not yet implemented.

## Model

XGBoost gradient-boosted classifier (300 trees, max depth 3, subsampling + L1/L2 regularization), with `scale_pos_weight` applied to counter class imbalance and prioritize recall on the at-risk class.

## Results

| Metric | Value |
|---|---|
| ROC-AUC (hold-out) | 0.777 |
| ROC-AUC (5-fold stratified CV) | 0.774 ± 0.005 |
| At-risk recall | 0.69 |
| At-risk precision | 0.26 |
| Accuracy | 0.73 |

The precision/recall tradeoff is intentional: for an early-warning system, a missed at-risk student (false negative) is more costly than a false alarm (false positive), so the model is tuned to favor recall. Raw accuracy is not a meaningful headline metric here — a model predicting "pass" for everyone would score ~88% accuracy while catching zero at-risk students; cross-validation was used to confirm the reported ROC-AUC is stable and not an artifact of one lucky train/test split.

Top predictive features: prior-year average mark, education level, socio-economic rank, and midterm score. Model-derived feature importances were cross-checked against raw point-biserial correlations with the target to confirm they reflect genuine signal rather than modeling artifacts.

## Limitations

- Built on a sample/synthetic dataset — patterns may not generalize to a real student population.
- Missing-value imputation uses a single global median rather than subgroup-aware imputation.
- Cross-validation does not yet account for repeated students across folds (see leakage safeguards above).
- No logistic regression or dummy-classifier baseline has been run yet to quantify how much predictive power comes from XGBoost's non-linear/interaction modeling versus the engineered features themselves.

## Requirements

```
pandas
numpy
scikit-learn
xgboost
```

## Usage

```bash
python build_risk_model.py
```

Runs the full pipeline end-to-end: loads and joins the data, engineers features, trains the model, and prints evaluation metrics, feature importances, and cross-validation results.
