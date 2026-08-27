import pandas as pd
import numpy as np

PASS_THRESHOLD = 40

df = pd.read_csv("model_dataset_sample20k.csv")

df["at_risk"] = (df["final_score"] < PASS_THRESHOLD).astype(int)

weeks = range(1, 7)
early_weeks = [1, 2, 3]
late_weeks = [4, 5, 6]

for w in weeks:
    attended = df[f"attended_sessions_w0{w}"]
    total = df[f"total_sessions_w0{w}"].replace(0, np.nan)
    df[f"attend_rate_w0{w}"] = attended / total

df["attend_rate_early"] = df[[f"attend_rate_w0{w}" for w in early_weeks]].mean(axis=1)
df["attend_rate_late"] = df[[f"attend_rate_w0{w}" for w in late_weeks]].mean(axis=1)
df["attend_rate_trend"] = df["attend_rate_late"] - df["attend_rate_early"]

vle_metrics = ["vle_logins", "vle_resource_views", "vle_forum_posts"]
for metric in vle_metrics:
    df[f"{metric}_early"] = df[[f"{metric}_w0{w}" for w in early_weeks]].sum(axis=1)
    df[f"{metric}_late"] = df[[f"{metric}_w0{w}" for w in late_weeks]].sum(axis=1)
    df[f"{metric}_trend"] = df[f"{metric}_late"] - df[f"{metric}_early"]

students = pd.read_csv("stonegrove-data/dim_students.csv")
survey = pd.read_csv("stonegrove-data/fact_enrolment_survey.csv")
progression = pd.read_csv("stonegrove-data/fact_progression.csv")

df = df.merge(
    students[["student_id", "age", "education", "socio_economic_rank",
              "disabilities", "first_gen"]],
    on="student_id", how="left"
)

survey_cols = ["student_id", "academic_year", "is_repeat_year", "career_clarity",
               "career_confidence", "belonging_peers", "belonging_programme",
               "academic_self_efficacy", "support_satisfaction"]
df = df.merge(survey[survey_cols], on=["student_id", "academic_year"], how="left")

prog_sorted = progression[["student_id", "academic_year", "avg_mark"]].sort_values(
    ["student_id", "academic_year"])
prog_sorted["prior_avg_mark"] = prog_sorted.groupby("student_id")["avg_mark"].shift(1)
df = df.merge(
    prog_sorted[["student_id", "academic_year", "prior_avg_mark"]],
    on=["student_id", "academic_year"], how="left"
)

modules = pd.read_csv("stonegrove-data/dim_modules.csv")
df = df.merge(modules[["module_code", "module_year", "assessment_type"]],
              on="module_code", how="left")

df["has_prior_year"] = df["prior_avg_mark"].notna().astype(int)
df["prior_avg_mark"] = df["prior_avg_mark"].fillna(df["prior_avg_mark"].median())

survey_numeric_cols = ["career_clarity", "career_confidence", "belonging_peers",
                       "belonging_programme", "academic_self_efficacy", "support_satisfaction"]
for col in survey_numeric_cols:
    df[col] = df[col].fillna(df[col].median())

all_tags = set()
for entry in df["disabilities"].dropna():
    all_tags.update(entry.split(","))
all_tags.discard("no_known_disabilities")

for tag in sorted(all_tags):
    df[f"disability_{tag}"] = df["disabilities"].fillna("").apply(
        lambda s: int(tag in s.split(","))
    )

df = df.drop(columns=["disabilities"])

categorical_cols = ["education", "assessment_type"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

non_feature_cols = {
    "student_id", "module_code", "academic_year", "semester",
    "final_score", "at_risk",
}
non_feature_cols.update(c for c in df.columns if c.startswith("attended_sessions_w0"))
non_feature_cols.update(c for c in df.columns if c.startswith("total_sessions_w0"))
non_feature_cols.update(c for c in df.columns if c.startswith("vle_") and c.split("_w0")[-1].isdigit())

features = [c for c in df.columns if c not in non_feature_cols]
target = "at_risk"

df_model = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()

print(f"Feature count: {len(features)}")
print(f"Rows after dropping remaining NaNs: {len(df_model)} (started at {len(df)})")

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fail_rate = y_train.mean()
scale_pos_weight = (1 - fail_rate) / fail_rate
print(f"\nTraining fail rate: {fail_rate*100:.1f}%  ->  scale_pos_weight = {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Pass", "At-Risk"]))

importance = (
    pd.Series(model.feature_importances_, index=features)
    .sort_values(ascending=False)
    .head(15)
)
print("Top 15 Feature Importances:")
for feat, score in importance.items():
    bar = "#" * int(score * 50)
    print(f"  {feat:<35} {score:.4f}  {bar}")

print("\nRaw correlation with at_risk (independent of the model):")
corr_check = [
    "midterm_score", "prior_avg_mark", "attend_rate_early", "attend_rate_late",
    "socio_economic_rank", "education_no_qualifications", "education_vocational",
]
correlations = df_model[corr_check + ["at_risk"]].corr()["at_risk"].drop("at_risk")
correlations = correlations.sort_values(key=abs, ascending=False)
for feat, val in correlations.items():
    direction = "higher value -> more risk" if val > 0 else "higher value -> less risk"
    print(f"  {feat:<30} {val:+.3f}  ({direction})")

from sklearn.model_selection import StratifiedKFold

print("\n5-Fold Stratified Cross-Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []

for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
    y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_fail_rate = y_fold_train.mean()
    fold_scale_pos_weight = (1 - fold_fail_rate) / fold_fail_rate

    fold_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, scale_pos_weight=fold_scale_pos_weight,
        eval_metric="auc",
    )
    fold_model.fit(X_fold_train, y_fold_train)

    fold_pred_prob = fold_model.predict_proba(X_fold_test)[:, 1]
    fold_auc = roc_auc_score(y_fold_test, fold_pred_prob)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold_num}: ROC-AUC = {fold_auc:.4f}")

fold_aucs = np.array(fold_aucs)
print(f"\nMean ROC-AUC: {fold_aucs.mean():.4f}  (std: {fold_aucs.std():.4f})")
