#!/usr/bin/env python3
"""Build cases_attrition_100.csv and cases_attrition_100_final.csv.

Usage (from repo root):
    python example_attrition/cases_attrition_100/build_cases_attrition_100.py

Requires: pandas, h2o, Java/H2O runtime, MOJOs in example_attrition/models/.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import h2o
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fuzzy_context import calculate_context_score  # noqa: E402

EXAMPLE_DIR = REPO_ROOT / "example_attrition"
OUT_DIR = EXAMPLE_DIR / "cases_attrition_100"
EXISTING_CASES = EXAMPLE_DIR / "data" / "cases_atttrition.csv"
IBM_REVISED = EXAMPLE_DIR / "dataset" / "IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"
CONTEXT_JSON = EXAMPLE_DIR / "context" / "context_digital_transformation.json"
MODEL_DEFS = EXAMPLE_DIR / "model_definitions.json"
MODELS_DIR = EXAMPLE_DIR / "models"

CASE_COLUMNS = [
    "Case_ID",
    "attrition",
    "age",
    "business_travel",
    "daily_rate",
    "department",
    "distance_from_home",
    "education",
    "education_field",
    "employee_count",
    "employee_number",
    "environment_satisfaction",
    "gender",
    "hourly_rate",
    "job_involvement",
    "job_level",
    "job_role",
    "job_satisfaction",
    "marital_status",
    "monthly_income",
    "monthly_rate",
    "num_companies_worked",
    "over18",
    "over_time",
    "percent_salary_hike",
    "performance_rating",
    "relationship_satisfaction",
    "standard_hours",
    "stock_option_level",
    "total_working_years",
    "training_times_last_year",
    "work_life_balance",
    "years_at_company",
    "years_in_current_role",
    "years_since_last_promotion",
    "years_with_curr_manager",
]

FINGERPRINT_COLS = [c for c in CASE_COLUMNS if c not in ("Case_ID", "employee_number")]

EDUCATION_MAP = {
    "Below College": 1,
    "College": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctor": 5,
}
JOB_LEVEL_MAP = {
    "Entry Level": 1,
    "Junior Level": 2,
    "Mid Level": 3,
    "Senior Level": 4,
    "Director": 5,
    "Executive Level": 5,
}
SAT4_MAP = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
WLB_MAP = {"Bad": 1, "Good": 2, "Better": 3, "Best": 4}
PERF_MAP = {"Low": 1, "Good": 2, "Excellent": 3, "Outstanding": 4}

ATOMIC_RULES = [
    "training_high",
    "education_high",
    "years_company_relevant",
    "years_role_relevant",
    "age_investment_window",
    "experience_sufficient",
    "role_adaptability",
]

MODEL_SHORT = {
    "cw_StackedEnsemble_BestOfFamily_4_AutoML_1_20260118_132607.zip": "cw",
    "fin_StackedEnsemble_BestOfFamily_4_AutoML_2_20260118_150802.zip": "fin",
    "od_GBM_grid_1_AutoML_3_20260118_163705_model_26.zip": "od",
    "hr_StackedEnsemble_BestOfFamily_4_AutoML_2_20260119_90304.zip": "hr",
}

FIRST_NAMES = [
    "Aaron", "Abigail", "Adam", "Adrian", "Aiden", "Alex", "Alice", "Amelia", "Andrew", "Anna",
    "Anthony", "Aria", "Arthur", "Austin", "Ava", "Barbara", "Benjamin", "Blake", "Brandon", "Brian",
    "Brooke", "Caleb", "Camila", "Carlos", "Caroline", "Catherine", "Charles", "Chloe", "Christian",
    "Christopher", "Claire", "Cole", "Daniel", "David", "Diana", "Dominic", "Dylan", "Edward",
    "Eleanor", "Elijah", "Elizabeth", "Emily", "Emma", "Ethan", "Eva", "Evan", "Faith", "Felix",
    "Finn", "Gabriel", "Grace", "Grant", "Hannah", "Harper", "Henry", "Hunter", "Ian", "Isaac",
    "Isabella", "Jack", "Jackson", "Jacob", "Jade", "James", "Jasmine", "Jason", "Jennifer", "Jessica",
    "John", "Jonathan", "Jordan", "Joseph", "Joshua", "Julia", "Justin", "Katherine", "Kevin", "Laura",
    "Lauren", "Leah", "Leo", "Liam", "Lily", "Logan", "Lucas", "Lucy", "Luke", "Madison",
    "Maria", "Mark", "Mason", "Matthew", "Maya", "Michael", "Mila", "Natalie", "Nathan", "Nicholas",
    "Nicole", "Noah", "Nora", "Oliver", "Olivia", "Oscar", "Owen", "Patrick", "Paul", "Penelope",
    "Peter", "Quinn", "Rachel", "Rebecca", "Richard", "Robert", "Ryan", "Samantha", "Samuel", "Sarah",
    "Scarlett", "Sean", "Sebastian", "Sophia", "Stella", "Stephanie", "Steven", "Thomas", "Timothy",
    "Tyler", "Victoria", "Vincent", "Violet", "William", "Zachary", "Zoe",
]

LAST_NAMES = [
    "Adams", "Allen", "Anderson", "Bailey", "Baker", "Barnes", "Bell", "Bennett", "Brooks", "Brown",
    "Butler", "Campbell", "Carter", "Clark", "Collins", "Cook", "Cooper", "Cox", "Cruz", "Davis",
    "Diaz", "Edwards", "Evans", "Fisher", "Flores", "Foster", "Garcia", "Gonzalez", "Gray", "Green",
    "Griffin", "Hall", "Hamilton", "Harris", "Hayes", "Hernandez", "Hill", "Howard", "Hughes", "Jackson",
    "James", "Jenkins", "Johnson", "Jones", "Kelly", "King", "Lee", "Lewis", "Long", "Lopez",
    "Martin", "Martinez", "Miller", "Mitchell", "Moore", "Morgan", "Morris", "Murphy", "Myers", "Nelson",
    "Nguyen", "Parker", "Patel", "Perez", "Perry", "Peterson", "Phillips", "Powell", "Price", "Ramirez",
    "Reed", "Reyes", "Richardson", "Rivera", "Roberts", "Robinson", "Rodriguez", "Rogers", "Ross", "Russell",
    "Sanchez", "Sanders", "Scott", "Simmons", "Smith", "Stewart", "Taylor", "Thomas", "Thompson", "Torres",
    "Turner", "Walker", "Ward", "Washington", "Watson", "White", "Williams", "Wilson", "Wood", "Wright",
    "Young",
]

LAMBDA = 0.5
LOW_CI_THRESHOLD = 0.15
SELECTION_QUOTA = {"low": 12, "mid": 18, "high": 10}
RANDOM_SEED = 42


def convert_ibm_row(df: pd.DataFrame) -> pd.DataFrame:
    """Map IBM Revised textual CSV to cases_atttrition schema."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["Age"].astype(int)
    out["attrition"] = df["Attrition"]
    out["business_travel"] = df["BusinessTravel"].replace({"Non-Travel": "Non_Travel"})
    out["daily_rate"] = df["DailyRate"].astype(int)
    out["department"] = df["Department"]
    out["distance_from_home"] = df["DistanceFromHome"].astype(int)
    out["education"] = df["Education"].map(EDUCATION_MAP)
    out["education_field"] = df["EducationField"]
    out["employee_count"] = 1
    out["environment_satisfaction"] = df["EnvironmentSatisfaction"].map(SAT4_MAP)
    out["gender"] = df["Gender"]
    out["hourly_rate"] = df["HourlyRate"].astype(int)
    out["job_involvement"] = df["JobInvolvement"].map(SAT4_MAP)
    out["job_level"] = df["JobLevel"].map(JOB_LEVEL_MAP)
    out["job_role"] = df["JobRole"]
    out["job_satisfaction"] = df["JobSatisfaction"].map(SAT4_MAP)
    out["marital_status"] = df["MaritalStatus"]
    out["monthly_income"] = df["MonthlyIncome"].astype(int)
    out["monthly_rate"] = df["MonthlyRate"].astype(int)
    out["num_companies_worked"] = df["NumCompaniesWorked"].astype(int)
    out["over18"] = "Y"
    out["over_time"] = df["OverTime"]
    out["percent_salary_hike"] = df["PercentSalaryHike"].astype(int)
    out["performance_rating"] = df["PerformanceRating"].map(PERF_MAP)
    out["relationship_satisfaction"] = df["RelationshipSatisfaction"].map(SAT4_MAP)
    out["standard_hours"] = 80
    out["stock_option_level"] = df["StockOptionLevel"].astype(int)
    out["total_working_years"] = df["TotalWorkingYears"].astype(int)
    out["training_times_last_year"] = df["TrainingTimesLastYear"].astype(int)
    out["work_life_balance"] = df["WorkLifeBalance"].map(WLB_MAP)
    out["years_at_company"] = df["YearsAtCompany"].astype(int)
    out["years_in_current_role"] = df["YearsInCurrentRole"].astype(int)
    out["years_since_last_promotion"] = df["YearsSinceLastPromotion"].astype(int)
    out["years_with_curr_manager"] = df["YearsWithCurrManager"].astype(int)
    return out


def fingerprint_frame(df: pd.DataFrame) -> pd.Series:
    return df[FINGERPRINT_COLS].astype(str).agg("|".join, axis=1)


def assign_case_ids(n: int, used: set[str], rng: random.Random) -> list[str]:
    pool = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(pool)
    ids: list[str] = []
    for name in pool:
        if name in used:
            continue
        ids.append(name)
        used.add(name)
        if len(ids) >= n:
            break
    if len(ids) < n:
        raise RuntimeError(f"Could not generate {n} unique Case_ID values.")
    return ids

def classify_context_band(ci: float) -> str:
    if ci <= LOW_CI_THRESHOLD:
        return "low"
    if ci <= 0.45:
        return "mid"
    return "high"


def select_stratified(
    subset: pd.DataFrame,
    ci_series: pd.Series,
    context_config: dict,
    n: int,
    rng: random.Random,
) -> pd.Index:
    """Pick n rows with ~12 low / 18 mid / 10 high Ci bands per attrition stratum."""
    work = subset.copy()
    work["_ci"] = ci_series.loc[work.index].values
    work["_band"] = work["_ci"].apply(classify_context_band)

    selected: list[int] = []
    for band, quota in SELECTION_QUOTA.items():
        band_df = work[(work["_band"] == band) & (~work.index.isin(selected))]
        if len(band_df) >= quota:
            picks = list(
                band_df.sample(n=quota, random_state=rng.randint(0, 10_000)).index
            )
        else:
            picks = list(band_df.index)
            still_need = quota - len(picks)
            if still_need > 0:
                remaining = work[~work.index.isin(selected + picks)]
                if band == "low" and len(band_df) == 0:
                    alt_ci, _ = calculate_context_score(
                        remaining, context_config, aggregation="minimum (strict)"
                    )
                    remaining = remaining.copy()
                    remaining["_ci_pick"] = alt_ci.values
                else:
                    remaining = remaining.copy()
                    remaining["_ci_pick"] = remaining["_ci"]
                extra = remaining.nsmallest(still_need, "_ci_pick")
                picks.extend(list(extra.index))
        selected.extend(picks[:quota])

    if len(selected) < n:
        remaining = work[~work.index.isin(selected)]
        extra = remaining.sample(
            n=n - len(selected), random_state=rng.randint(0, 10_000)
        )
        selected.extend(list(extra.index))

    return pd.Index(selected[:n])


def validate_outputs(
    existing: pd.DataFrame,
    cases_100: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    """Post-generation checks from the dataset plan."""
    feature_cols = [c for c in existing.columns if c != "Case_ID"]
    if not (existing[feature_cols].values == cases_100.iloc[:20][feature_cols].values).all():
        raise AssertionError("First 20 rows must match cases_atttrition.csv unchanged.")

    fps = fingerprint_frame(cases_100)
    if fps.duplicated().any():
        raise AssertionError("Duplicate case fingerprints detected.")

    new_only = cases_100.iloc[20:]
    if (new_only["attrition"] == "Yes").sum() != 40 or (new_only["attrition"] == "No").sum() != 40:
        raise AssertionError("New 80 cases must be 40 Yes / 40 No attrition.")

    for col in ("Ri_Global_Risk", "Ci_Context_Score", "Prioritization_Score"):
        if final[col].min() < 0 or final[col].max() > 1:
            raise AssertionError(f"{col} out of [0, 1] range.")

    low_ci = (final["Ci_Context_Score"] <= LOW_CI_THRESHOLD).sum()
    if low_ci < 20:
        raise AssertionError(f"Expected >= 20 low-context cases (Ci <= {LOW_CI_THRESHOLD}), got {low_ci}.")

    if len(cases_100) != 100 or len(final) != 100:
        raise AssertionError("Output must contain exactly 100 rows.")


def init_h2o() -> None:
    h2o.init(max_mem_size="700m", nthreads=1)


def run_mojo_predictions(df: pd.DataFrame, feature_config: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    model_df = df.drop(columns=["Case_ID"], errors="ignore")
    weights_raw = {m: feature_config[m]["performance"]["auc"] for m in feature_config}
    total = sum(weights_raw.values())
    weights = {m: v / total for m, v in weights_raw.items()}

    probs = pd.DataFrame(index=df.index)
    for model_file, short in MODEL_SHORT.items():
        path = str(MODELS_DIR / model_file)
        mojo = h2o.import_mojo(path)
        hf = h2o.H2OFrame(model_df)
        try:
            preds = mojo.predict(hf).as_data_frame()
            p_col = "p1" if "p1" in preds.columns else preds.columns[-1]
            probs[f"{short}_prob"] = preds[p_col].values
        finally:
            h2o.remove(hf)

    risk = np.zeros(len(df))
    for model_file, short in MODEL_SHORT.items():
        risk += probs[f"{short}_prob"].values * weights[model_file]

    probs["Ri_Global_Risk"] = risk
    return probs, weights


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(EXISTING_CASES, sep=";")
    existing = existing[CASE_COLUMNS]

    ibm = pd.read_csv(IBM_REVISED)
    converted = convert_ibm_row(ibm)

    unmapped = converted.isna().any(axis=1)
    if unmapped.any():
        bad = converted[unmapped]
        raise ValueError(f"Unmapped IBM values in {unmapped.sum()} rows:\n{bad.head()}")

    existing_fp = set(fingerprint_frame(existing))
    converted["_fp"] = fingerprint_frame(converted)
    pool = converted[~converted["_fp"].isin(existing_fp)].copy()
    pool = pool.drop(columns=["_fp"])

    with open(CONTEXT_JSON, encoding="utf-8") as f:
        context_config = json.load(f)

    ci_all, fuzzy_all = calculate_context_score(pool, context_config, aggregation=None)
    pool["_ci"] = ci_all.values

    used_ids = set(existing["Case_ID"].astype(str))
    new_frames: list[pd.DataFrame] = []

    for attrition_label, n_pick in [("Yes", 40), ("No", 40)]:
        subset = pool[pool["attrition"] == attrition_label]
        picked_idx = select_stratified(subset, ci_all, context_config, n_pick, rng)
        chunk = subset.loc[picked_idx].copy()
        names = assign_case_ids(len(chunk), used_ids, rng)
        chunk.insert(0, "Case_ID", names)
        new_frames.append(chunk)

    new_80 = pd.concat(new_frames, ignore_index=True)
    new_80["employee_number"] = range(2000, 2000 + len(new_80))
    new_80 = new_80[CASE_COLUMNS]
    cases_100 = pd.concat([existing, new_80], ignore_index=True)

    cases_path = OUT_DIR / "cases_attrition_100.csv"
    cases_100.to_csv(cases_path, sep=";", index=False)
    print(f"Wrote {cases_path} ({len(cases_100)} rows)")

    init_h2o()
    with open(MODEL_DEFS, encoding="utf-8") as f:
        feature_config = json.load(f)

    model_probs, weights = run_mojo_predictions(cases_100, feature_config)
    ci_scores, fuzzy_df = calculate_context_score(cases_100, context_config, aggregation=None)
    prioritization = LAMBDA * model_probs["Ri_Global_Risk"] + (1 - LAMBDA) * ci_scores.values

    atomic_cols = [f"mu_{r}" for r in ATOMIC_RULES]
    final = pd.DataFrame({
        "Case_ID": cases_100["Case_ID"],
        "attrition": cases_100["attrition"],
        "Ri_Global_Risk": model_probs["Ri_Global_Risk"].round(6),
        "Ci_Context_Score": ci_scores.round(6),
        "Prioritization_Score": np.round(prioritization, 6),
        "cw_prob": model_probs["cw_prob"].round(6),
        "fin_prob": model_probs["fin_prob"].round(6),
        "od_prob": model_probs["od_prob"].round(6),
        "hr_prob": model_probs["hr_prob"].round(6),
    })
    for col in atomic_cols:
        final[col] = fuzzy_df[col].round(6)

    final_path = OUT_DIR / "cases_attrition_100_final.csv"
    final.to_csv(final_path, sep=";", index=False)
    print(f"Wrote {final_path} ({len(final)} rows)")

    validate_outputs(existing, cases_100, final)

    # Validation summary
    new_only = cases_100.iloc[20:]
    print("\n--- Validation ---")
    print(f"Total rows: {len(cases_100)}")
    print(f"New attrition Yes/No: {(new_only['attrition'] == 'Yes').sum()}/{(new_only['attrition'] == 'No').sum()}")
    print(f"Ci <= {LOW_CI_THRESHOLD} (all 100): {(final['Ci_Context_Score'] <= LOW_CI_THRESHOLD).sum()}")
    print(f"Ci <= {LOW_CI_THRESHOLD} (new 80): {(final.iloc[20:]['Ci_Context_Score'] <= LOW_CI_THRESHOLD).sum()}")
    print(f"Ri range: [{final['Ri_Global_Risk'].min():.4f}, {final['Ri_Global_Risk'].max():.4f}]")
    print(f"Model weights (AUC): { {MODEL_SHORT[k]: round(v, 4) for k, v in weights.items()} }")

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
