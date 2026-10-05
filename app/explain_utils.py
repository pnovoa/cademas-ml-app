"""Shared helpers for the Explain (XAI) tab."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

FEATURE_ALIASES = {
    "over_18": "over18",
}

FEATURE_LABELS = {
    "attrition": "Attrition",
    "age": "Age",
    "business_travel": "Business travel",
    "daily_rate": "Daily rate",
    "department": "Department",
    "distance_from_home": "Distance from home",
    "education": "Education",
    "education_field": "Education field",
    "employee_count": "Employee count",
    "employee_number": "Employee number",
    "environment_satisfaction": "Environment satisfaction",
    "gender": "Gender",
    "hourly_rate": "Hourly rate",
    "job_involvement": "Job involvement",
    "job_level": "Job level",
    "job_role": "Job role",
    "job_satisfaction": "Job satisfaction",
    "marital_status": "Marital status",
    "monthly_income": "Monthly income",
    "monthly_rate": "Monthly rate",
    "num_companies_worked": "Number of companies worked",
    "over18": "Over 18",
    "over_18": "Over 18",
    "over_time": "Overtime",
    "percent_salary_hike": "Percent salary hike",
    "performance_rating": "Performance rating",
    "relationship_satisfaction": "Relationship satisfaction",
    "standard_hours": "Standard hours",
    "stock_option_level": "Stock option level",
    "total_working_years": "Total working years",
    "training_times_last_year": "Training times last year",
    "work_life_balance": "Work-life balance",
    "years_at_company": "Years at company",
    "years_in_current_role": "Years in current role",
    "years_since_last_promotion": "Years since last promotion",
    "years_with_curr_manager": "Years with current manager",
}


def resolve_feature_column(feature_name: str, columns: list[str] | pd.Index) -> str | None:
    """Map JSON feature names to dataset column names."""
    cols = list(columns)
    if feature_name in cols:
        return feature_name
    alias = FEATURE_ALIASES.get(feature_name)
    if alias and alias in cols:
        return alias
    return None


def humanize_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    label = name.replace("_", " ")
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    return label[:1].upper() + label[1:]


def humanize_rule(name: str, context_config: dict | None = None) -> str:
    """Return a display label for a context rule, preferring JSON full_name."""
    if context_config:
        for rule in context_config.get("rules", []) + context_config.get("derived_rules", []):
            if rule.get("name") == name:
                return rule.get("full_name") or humanize_feature(name)
    return humanize_feature(name)


def humanize_membership_column(column: str, context_config: dict | None = None) -> str:
    """Humanize a fuzzy audit column such as mu_tenure_relevant."""
    if str(column).startswith("mu_"):
        return humanize_rule(str(column)[3:], context_config)
    return humanize_feature(str(column))


def humanize_value(value: Any) -> str:
    text = str(value)
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    if text == "Y":
        return "Yes"
    if text == "N":
        return "No"
    return text


def get_case_row_index(case_id: Any, df: pd.DataFrame, id_col: str = "CaseID") -> int | None:
    matches = df.index[df[id_col] == case_id].tolist()
    if not matches:
        return None
    return int(matches[0])


def top_contributions(series: pd.Series, k: int = 5) -> pd.Series:
    """Return top-k features by absolute contribution, preserving sign."""
    if series.empty:
        return series
    ranked = series.reindex(series.abs().sort_values(ascending=False).index)
    return ranked.head(k)


def aggregate_weighted_attributions(
    by_model: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Weighted sum of per-model attribution matrices."""
    if not by_model:
        return pd.DataFrame()

    all_features: list[str] = []
    seen = set()
    for frame in by_model.values():
        for col in frame.columns:
            if col not in seen:
                seen.add(col)
                all_features.append(col)

    n_rows = next(iter(by_model.values())).shape[0]
    aggregated = pd.DataFrame(0.0, index=range(n_rows), columns=all_features)

    for model_name, frame in by_model.items():
        weight = float(weights.get(model_name, 0.0))
        if weight == 0:
            continue
        for feature in frame.columns:
            aggregated[feature] = aggregated[feature] + weight * frame[feature].values

    return aggregated


def _numeric_membership_columns(fuzzy_row: pd.Series) -> pd.Series:
    numeric = {}
    for col, val in fuzzy_row.items():
        if not str(col).startswith("mu_"):
            continue
        if isinstance(val, (int, float, np.floating, np.integer)):
            numeric[str(col)[3:]] = float(val)
    return pd.Series(numeric)


def bottleneck_rule(
    fuzzy_row: pd.Series,
    aggregation_method: str,
    context_config: dict | None = None,
) -> tuple[str, float] | None:
    """Return the limiting rule name and membership for minimum aggregation."""
    series = _numeric_membership_columns(fuzzy_row)
    if series.empty:
        return None

    if aggregation_method == "minimum (strict)":
        rule_name = series.idxmin()
        return rule_name, float(series.min())

    if context_config and context_config.get("logic"):
        rule_name = series.idxmin()
        return rule_name, float(series.min())

    return None


def split_context_rules(
    fuzzy_row: pd.Series,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Split atomic rule memberships into contextual drivers and penalizers."""
    series = _numeric_membership_columns(fuzzy_row)
    drivers = [(name, float(val)) for name, val in series.items() if val > 0.6]
    bottlenecks = [(name, float(val)) for name, val in series.items() if val < 0.4]
    drivers.sort(key=lambda item: item[1], reverse=True)
    bottlenecks.sort(key=lambda item: item[1])
    return drivers, bottlenecks


def classify_risk_level(ri: float) -> str:
    if ri >= 0.8:
        return "critical"
    if ri >= 0.5:
        return "high"
    return "low"


def classify_context_level(ci: float) -> str:
    if ci >= 0.7:
        return "high"
    if ci >= 0.4:
        return "moderate"
    return "low"


def classify_balance_stance(lambda_val: float, tolerance: float = 0.05) -> str:
    if abs(lambda_val - 0.5) <= tolerance:
        return "balanced"
    if lambda_val > 0.5:
        return "predictive-risk-led"
    return "context-policy-led"


def _format_balance_phrase(lambda_val: float) -> str:
    stance = classify_balance_stance(lambda_val)
    if stance == "balanced":
        return "balanced"
    if stance == "predictive-risk-led":
        return "predictive-risk-led"
    return "context-policy-led"


def _format_modulation_phrase(
    modulation_operator: str,
    lambda_val: float,
    ri: float,
    ci: float,
) -> str:
    """Describe how d and c were combined into the priority score p for the NL summary."""
    from modulation import (
        OPERATOR_GEOMETRIC,
        OPERATOR_LINEAR,
        normalize_operator,
    )

    op = normalize_operator(modulation_operator)
    if op == OPERATOR_LINEAR:
        balance_phrase = _format_balance_phrase(lambda_val)
        return (
            f"a {balance_phrase} Linear modulation posture (λ={lambda_val:.2f}), "
            f"with p = λ·d + (1−λ)·c"
        )
    if op == OPERATOR_GEOMETRIC:
        balance_phrase = _format_balance_phrase(lambda_val)
        return (
            f"a {balance_phrase} Geometric modulation posture (λ={lambda_val:.2f}), "
            f"with p = d^λ · c^(1−λ)"
        )
    limiting = (
        "the integrated predictive score (d)"
        if ri <= ci
        else "the contextual alignment score (c)"
    )
    if abs(ri - ci) < 1e-12:
        limiting = "both components equally"
    return (
        f"a Minimum modulation (parameter-free), "
        f"with p = min{{d, c}} limited by {limiting}"
    )


DEFAULT_NLG_VOCABULARY: dict[str, str] = {
    "target_case": "the case",
    "target_action": "intervention or resource allocation",
}


def _resolve_nlg_vocabulary(vocabulary: dict[str, str] | None = None) -> dict[str, str]:
    """Merge caller overrides with the default domain-agnostic NLG vocabulary."""
    resolved = DEFAULT_NLG_VOCABULARY.copy()
    if vocabulary:
        resolved.update(vocabulary)
    return resolved


def _bold_label(text: str) -> str:
    """Wrap a display label in markdown bold for NLG output."""
    return f"**{text}**"


def _format_ml_driver_phrase(top_ml: pd.Series, master_row: pd.Series, k: int = 2) -> str:
    ranked = top_contributions(top_ml, k=k)
    parts = []
    for feature in ranked.index:
        col = resolve_feature_column(feature, master_row.index)
        raw_val = master_row[col] if col is not None and col in master_row.index else "n/a"
        parts.append(
            f"{_bold_label(humanize_feature(feature))} ({humanize_value(raw_val)})"
        )
    if not parts:
        return "no single feature dominates the predictive signal"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} and {parts[1]}"


def _format_context_sentence(
    ci: float,
    fuzzy_row: pd.Series,
    context_config: dict | None = None,
) -> str:
    drivers, bottlenecks = split_context_rules(fuzzy_row)

    if ci > 0.5:
        if len(drivers) >= 2:
            return (
                "aligns strongly with the policy rules, "
                f"particularly on {_bold_label(humanize_rule(drivers[0][0], context_config))} and "
                f"{_bold_label(humanize_rule(drivers[1][0], context_config))}"
            )
        if len(drivers) == 1:
            return (
                "aligns with the policy rules, "
                f"with {_bold_label(humanize_rule(drivers[0][0], context_config))} standing out as a contextual strength"
            )
        return (
            "shows positive contextual alignment overall, although no individual policy rule "
            "reaches a strong driver threshold (μ > 0.6)"
        )

    if len(bottlenecks) >= 2:
        return (
            "does not meet the policy criteria for "
            f"{_bold_label(humanize_rule(bottlenecks[0][0], context_config))} (μ={bottlenecks[0][1]:.2f}) or "
            f"{_bold_label(humanize_rule(bottlenecks[1][0], context_config))} (μ={bottlenecks[1][1]:.2f}), "
            "which constrains case priority"
        )
    if len(bottlenecks) == 1:
        return (
            "does not meet the policy criteria for "
            f"{_bold_label(humanize_rule(bottlenecks[0][0], context_config))} (μ={bottlenecks[0][1]:.2f}), "
            "which mainly constrains case priority"
        )
    return (
        "shows moderate contextual alignment, without a clear penalizing policy rule below "
        "the 0.4 threshold"
    )


def context_highlight_rules(
    fuzzy_row: pd.Series,
    ci: float,
    aggregation_method: str | None = None,
    context_config: dict | None = None,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return key contextual rules to show in the brief overview table."""
    drivers, bottlenecks = split_context_rules(fuzzy_row)
    if ci > 0.5 and drivers:
        return drivers[:k]
    if bottlenecks:
        return bottlenecks[:k]

    if aggregation_method == "minimum (strict)" or (
        context_config and context_config.get("logic")
    ):
        bottleneck = bottleneck_rule(fuzzy_row, aggregation_method or "", context_config)
        if bottleneck:
            return [bottleneck]

    series = _numeric_membership_columns(fuzzy_row)
    ranked = series.sort_values(ascending=ci <= 0.5)
    return [(idx, float(val)) for idx, val in ranked.head(k).items()]


def format_priority_verdict(
    priority: float,
    vocabulary: dict[str, str] | None = None,
) -> str:
    """Return a tier-based action recommendation from the final priority score."""
    vocab = _resolve_nlg_vocabulary(vocabulary)
    target_action = vocab["target_action"]

    if priority >= 0.75:
        return (
            "**Suggestion [Tier 1 - High Priority]:** "
            f"Recommended to proceed with {target_action} immediately."
        )
    if priority >= 0.50:
        return (
            "**Suggestion [Tier 2 - Medium Priority]:** "
            "Mixed or borderline evidence. Manual qualitative evaluation is required "
            "before proceeding."
        )
    return (
        "**Suggestion [Tier 3 - Low Priority]:** "
        f"{target_action.capitalize()} is not justified under current prediction and "
        "context guidelines."
    )


def build_nl_summary(
    case_id: Any,
    priority: float,
    ri: float,
    ci: float,
    lambda_val: float,
    top_ml: pd.Series,
    master_row: pd.Series,
    fuzzy_row: pd.Series,
    vocabulary: dict[str, str] | None = None,
    context_config: dict | None = None,
    modulation_operator: str = "linear",
) -> str:
    """Deterministic, domain-agnostic analyst-style summary for a selected case."""
    vocab = _resolve_nlg_vocabulary(vocabulary)
    target_case = vocab["target_case"]

    risk_level = classify_risk_level(ri)
    context_level = classify_context_level(ci)
    modulation_phrase = _format_modulation_phrase(
        modulation_operator, lambda_val, ri, ci
    )
    ml_phrase = _format_ml_driver_phrase(top_ml, master_row)
    context_phrase = _format_context_sentence(ci, fuzzy_row, context_config)
    verdict = format_priority_verdict(priority, vocab)

    body = (
        f"The case priority score for {_bold_label(str(case_id))} is "
        f"{_bold_label(f'{priority:.0%}')}. "
        f"This decision reflects {modulation_phrase}, "
        f"combining {risk_level} integrated predictive evidence ({ri:.0%}) with "
        f"{context_level} contextual alignment ({ci:.0%}). "
        f"On the predictive side, the signal was driven mainly by {ml_phrase}. "
        f"In context, {target_case} {context_phrase}."
    )
    return f"{body}\n\n{verdict}"
