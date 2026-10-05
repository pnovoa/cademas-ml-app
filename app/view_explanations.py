"""Explain (XAI) tab for CADEMAS-ML."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from explain_utils import (
    bottleneck_rule,
    build_nl_summary,
    get_case_row_index,
    humanize_feature,
    humanize_membership_column,
    humanize_rule,
    humanize_value,
    resolve_feature_column,
    top_contributions,
)
from modulation import (
    OPERATOR_GEOMETRIC,
    OPERATOR_LINEAR,
    modulate,
    normalize_operator,
)


def _priority_column() -> str:
    return "Prioritization_Score "


ML_ATTRIBUTION_HELP = """
Contributions show how each feature shifts the model's predicted probability for the selected case relative to a cohort baseline.

**How they are calculated:** For each feature, the case value is temporarily replaced by the cohort baseline (median for numeric features, most frequent category for categorical ones). The contribution is the difference between the original prediction and the perturbed prediction: Δ = p(case) − p(case with feature at baseline). Per-model values are then combined using each model's weight w_k in the integrated predictive score.

**Units:** Probability points on a 0–1 scale (dimensionless). A value of +0.05 means the feature's actual value increases the model probability by about 5 percentage points compared with the baseline.

**How to read them for this case:** Red bars indicate features that push the integrated predictive score up; green bars push it down. Larger magnitudes mark stronger local drivers among the top contributors. These are approximate local effects (not causal impacts) and should be read together with the integrated predictive score (d) and contextual alignment (c).
"""


def _priority_summary_alert(summary: str, priority: float) -> None:
    """Render the NL summary in a traffic-light alert keyed to priority tier."""
    if priority >= 0.75:
        st.error(summary)
    elif priority >= 0.50:
        st.warning(summary)
    else:
        st.success(summary)


def _subheader_with_help(title: str, help_text: str) -> None:
    """Section subheader with a Streamlit-style help icon; click opens formatted help."""
    title_col, help_col = st.columns([0.96, 0.04], vertical_alignment="center")
    with title_col:
        st.subheader(title)
    with help_col:
        with st.popover(
            label="",
            icon=":material/help_outline:",
            type="tertiary",
        ):
            st.markdown(help_text)


def _count_ml_features(ml_attributions: dict) -> int:
    aggregated = ml_attributions.get("aggregated")
    if aggregated is None or aggregated.empty:
        return 0
    return int(len(aggregated.columns))


def _horizontal_bar_chart(
    chart_df: pd.DataFrame,
    y_field: str,
    x_field: str,
    x_title: str,
    color_condition: alt.condition | None = None,
    color_value: str = "#1E88E5",
    tooltip: list | None = None,
    height: int | None = None,
    x_scale: alt.Scale | None = None,
) -> alt.Chart:
    n_rows = len(chart_df)
    chart_height = height or max(220, 34 * n_rows)
    x_encoding = alt.X(f"{x_field}:Q", title=x_title)
    if x_scale is not None:
        x_encoding = alt.X(f"{x_field}:Q", title=x_title, scale=x_scale)

    encoding = {
        "y": alt.Y(
            f"{y_field}:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelLimit=700, labelOverlap=False, labelPadding=6),
        ),
        "x": x_encoding,
        "tooltip": tooltip or [alt.Tooltip(f"{y_field}:N"), alt.Tooltip(f"{x_field}:Q", format=".4f")],
    }
    if color_condition is not None:
        encoding["color"] = color_condition
    else:
        encoding["color"] = alt.value(color_value)

    return alt.Chart(chart_df).mark_bar().encode(**encoding).properties(height=chart_height)


def _contribution_chart_df(
    contributions: pd.Series,
    master_row: pd.Series,
    k: int = 5,
) -> pd.DataFrame:
    top = top_contributions(contributions, k=k)
    rows = []
    for feature, value in top.items():
        col = resolve_feature_column(feature, master_row.index)
        raw_val = master_row[col] if col and col in master_row.index else "n/a"
        rows.append(
            {
                "Feature": humanize_feature(feature),
                "Case value": humanize_value(raw_val),
                "Contribution": float(value),
            }
        )
    return pd.DataFrame(rows)


def _model_display_name(model_name: str, feature_config: dict | None) -> str:
    if feature_config and model_name in feature_config:
        department = feature_config[model_name].get("department")
        if department:
            return str(department)
    stem = model_name.split(".")[0]
    return stem.replace("_", " ")


def _render_level1(
    case_row: pd.Series,
    lambda_val: float,
    modulation_operator: str,
) -> None:
    st.subheader("Priority score breakdown")
    op = normalize_operator(modulation_operator)
    priority = float(case_row[_priority_column()])
    ri = float(case_row["Ri_Global_Risk"])
    ci = float(case_row["Ci_Context_Score"])

    if op == OPERATOR_LINEAR:
        ml_part = lambda_val * ri
        ctx_part = (1 - lambda_val) * ci

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Priority score (p)", f"{priority:.1%}")
        c2.metric("Integrated predictive (d)", f"{ri:.1%}")
        c3.metric("Contextual alignment (c)", f"{ci:.1%}")
        c4.metric("Lambda (λ)", f"{lambda_val:.2f}")

        breakdown = pd.DataFrame(
            {
                "Component": [
                    "Predictive contribution\n(λ·d)",
                    "Contextual contribution\n((1−λ)·c)",
                ],
                "Value": [ml_part, ctx_part],
                "Share": [
                    ml_part / priority if priority else 0,
                    ctx_part / priority if priority else 0,
                ],
            }
        )

        chart = (
            alt.Chart(breakdown)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Component:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelLimit=700,
                        labelOverlap=False,
                        labelPadding=8,
                    ),
                ),
                x=alt.X("Value:Q", title="Contribution to priority score"),
                color=alt.Color(
                    "Component:N",
                    scale=alt.Scale(range=["#1E88E5", "#43A047"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Component:N"),
                    alt.Tooltip("Value:Q", format=".3f"),
                    alt.Tooltip("Share:Q", format=".1%"),
                ],
            )
            .properties(height=150)
            .configure_view(strokeWidth=0, clip=False)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Linear: p = λ·d + (1−λ)·c = "
            f"{lambda_val:.2f}×{ri:.3f} + {1 - lambda_val:.2f}×{ci:.3f} = {priority:.3f}"
        )
        return

    if op == OPERATOR_GEOMETRIC:
        ri_factor = ri ** lambda_val
        ci_factor = ci ** (1 - lambda_val)
        recomputed = float(modulate(ri, ci, op, lambda_val))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Priority score (p)", f"{priority:.1%}")
        c2.metric("Integrated predictive (d)", f"{ri:.1%}")
        c3.metric("Contextual alignment (c)", f"{ci:.1%}")
        c4.metric("Lambda (λ)", f"{lambda_val:.2f}")

        factors = pd.DataFrame(
            {
                "Factor": [
                    f"d^λ = {ri:.3f}^{lambda_val:.2f}",
                    f"c^(1−λ) = {ci:.3f}^{1 - lambda_val:.2f}",
                ],
                "Value": [ri_factor, ci_factor],
            }
        )
        chart = (
            alt.Chart(factors)
            .mark_bar()
            .encode(
                y=alt.Y("Factor:N", title=None, sort="-x"),
                x=alt.X("Value:Q", title="Multiplicative factor"),
                color=alt.Color(
                    "Factor:N",
                    scale=alt.Scale(range=["#1E88E5", "#43A047"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Factor:N"),
                    alt.Tooltip("Value:Q", format=".4f"),
                ],
            )
            .properties(height=150)
            .configure_view(strokeWidth=0, clip=False)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Geometric: p = d^λ · c^(1−λ) = "
            f"{ri_factor:.4f} × {ci_factor:.4f} = {recomputed:.3f}"
        )
        st.info(
            "Under the geometric operator the combination is multiplicative, "
            "not an additive split of p into λ·d and (1−λ)·c."
        )
        return

    # Minimum
    limiting = (
        "Integrated predictive score (d)"
        if ri <= ci
        else "Contextual alignment score (c)"
    )
    if abs(ri - ci) < 1e-12:
        limiting = "Both (d = c)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Priority score (p)", f"{priority:.1%}")
    c2.metric("Integrated predictive (d)", f"{ri:.1%}")
    c3.metric("Contextual alignment (c)", f"{ci:.1%}")
    c4.metric("Lambda (λ)", "N/A")

    comparison = pd.DataFrame(
        {
            "Component": [
                "Integrated predictive score (d)",
                "Contextual alignment score (c)",
            ],
            "Value": [ri, ci],
            "Role": [
                "Limiting factor" if ri <= ci else "Non-binding",
                "Limiting factor" if ci <= ri else "Non-binding",
            ],
        }
    )
    chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            y=alt.Y("Component:N", title=None, sort="-x"),
            x=alt.X("Value:Q", title="Score"),
            color=alt.Color(
                "Role:N",
                scale=alt.Scale(
                    domain=["Limiting factor", "Non-binding"],
                    range=["#E53935", "#90A4AE"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("Component:N"),
                alt.Tooltip("Value:Q", format=".3f"),
                alt.Tooltip("Role:N"),
            ],
        )
        .properties(height=150)
        .configure_view(strokeWidth=0, clip=False)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Minimum: p = min{{d, c}} = min{{{ri:.3f}, {ci:.3f}}} = {priority:.3f}. "
        f"Limiting factor: {limiting}."
    )


def _render_level2(
    case_idx: int,
    case_row: pd.Series,
    master_row: pd.Series,
    ml_attributions: dict,
    ml_details: dict,
    feature_config: dict | None,
) -> None:
    _subheader_with_help("Predictive attribution", ML_ATTRIBUTION_HELP)
    aggregated = ml_attributions.get("aggregated")
    if aggregated is None or aggregated.empty:
        st.info("ML attributions are not available for this run.")
        return

    max_features = max(_count_ml_features(ml_attributions), 1)
    min_features = min(2, max_features)
    default_features = min(5, max_features)
    detail_k = st.slider(
        "Features to show",
        min_value=min_features,
        max_value=max_features,
        value=default_features,
        help="Number of top ML features shown in the attribution charts below.",
        key="explain_ml_feature_count",
    )

    contrib = aggregated.loc[case_idx]
    chart_df = _contribution_chart_df(contrib, master_row, k=detail_k)
    if chart_df.empty:
        st.info("No feature contributions were computed for this case.")
        return

    chart = _horizontal_bar_chart(
        chart_df,
        y_field="Feature",
        x_field="Contribution",
        x_title="Weighted contribution to integrated predictive score",
        color_condition=alt.condition(
            alt.datum.Contribution > 0,
            alt.value("#e13f40"),
            alt.value("#43A047"),
        ),
        tooltip=[
            alt.Tooltip("Feature:N"),
            alt.Tooltip("Case value:N"),
            alt.Tooltip("Contribution:Q", format="+.4f"),
        ],
    )
    st.altair_chart(chart, use_container_width=True)

    meta = ml_attributions.get("meta", {})
    st.caption(
        "Attribution method: "
        f"{meta.get('method', 'perturbation')} "
        f"(baseline: {meta.get('baseline', 'cohort')}). "
        "Contributions are approximate local effects aggregated across MOJO models."
    )

    with st.expander("Per-model breakdown"):
        by_model = ml_attributions.get("by_model", {})
        weights = (ml_details or {}).get("weights", {})
        for model_name, frame in by_model.items():
            if frame.empty:
                continue
            weight = weights.get(model_name, 0.0)
            model_df = _contribution_chart_df(frame.loc[case_idx], master_row, k=detail_k)
            if model_df.empty:
                continue
            st.markdown(
                f"**{_model_display_name(model_name, feature_config)}** "
                f"(weight {weight:.1%})"
            )
            model_chart = _horizontal_bar_chart(
                model_df,
                y_field="Feature",
                x_field="Contribution",
                x_title="Contribution to model probability",
                color_condition=alt.condition(
                    alt.datum.Contribution > 0,
                    alt.value("#e13f40"),
                    alt.value("#43A047"),
                ),
                tooltip=[
                    alt.Tooltip("Feature:N"),
                    alt.Tooltip("Case value:N"),
                    alt.Tooltip("Contribution:Q", format="+.4f"),
                ],
                height=max(180, 34 * len(model_df)),
            )
            st.altair_chart(model_chart, use_container_width=True)


def _render_level3(
    case_idx: int,
    fuzzy_row: pd.Series,
    aggregation_method: str,
    context_config: dict,
) -> None:
    st.subheader("Fuzzy context traceability")

    if context_config.get("logic"):
        st.info(
            "Contextual alignment score (c) was computed from the declarative logic tree in the context JSON. "
            "The sidebar aggregation operator is ignored when logic is present."
        )

    rule_rows = []
    for col, val in fuzzy_row.items():
        if not str(col).startswith("mu_"):
            continue
        if isinstance(val, (int, float)):
            rule_rows.append(
                {
                    "Rule": humanize_membership_column(str(col), context_config),
                    "Membership": float(val),
                }
            )
    if not rule_rows:
        st.info("No fuzzy membership values available for this case.")
        return

    rules_df = pd.DataFrame(rule_rows).sort_values("Membership")
    bottleneck = bottleneck_rule(fuzzy_row, aggregation_method, context_config)

    if aggregation_method == "minimum (strict)" and bottleneck:
        rule_name, rule_val = bottleneck
        st.warning(
            f"Bottleneck rule: **{humanize_rule(rule_name, context_config)}** "
            f"(μ = {rule_val:.2f}) defines the conservative context score."
        )

    chart = _horizontal_bar_chart(
        rules_df,
        y_field="Rule",
        x_field="Membership",
        x_title="Membership μ",
        color_condition=alt.condition(
            alt.datum.Membership < 0.35,
            alt.value("#e13f40"),
            alt.value("#1E88E5"),
        ),
        tooltip=[alt.Tooltip("Rule:N"), alt.Tooltip("Membership:Q", format=".3f")],
        height=max(220, 34 * len(rules_df)),
        x_scale=alt.Scale(domain=[0, 1]),
    )
    st.altair_chart(chart, use_container_width=True)


def _render_level4(
    case_id,
    case_idx: int,
    case_row: pd.Series,
    master_row: pd.Series,
    ml_attributions: dict,
    fuzzy_row: pd.Series,
    lambda_val: float,
    context_config: dict,
    modulation_operator: str,
) -> None:
    st.subheader("General explanation")
    aggregated = ml_attributions.get("aggregated")
    top_ml = (
        top_contributions(aggregated.loc[case_idx], k=5)
        if aggregated is not None and not aggregated.empty
        else pd.Series(dtype=float)
    )
    summary = build_nl_summary(
        case_id=case_id,
        priority=float(case_row[_priority_column()]),
        ri=float(case_row["Ri_Global_Risk"]),
        ci=float(case_row["Ci_Context_Score"]),
        lambda_val=lambda_val,
        top_ml=top_ml,
        master_row=master_row,
        fuzzy_row=fuzzy_row,
        context_config=context_config,
        modulation_operator=modulation_operator,
    )
    _priority_summary_alert(summary, float(case_row[_priority_column()]))


def render_explain_tab(
    df: pd.DataFrame,
    lambda_val: float,
    modulation_operator: str = "linear",
) -> None:
    """Render the Explain tab using precomputed session_state artifacts."""
    st.markdown(
        """
        This view explains **why** a case received its priority score.
        All explanations are read from the completed analysis run; no additional
        model inference is performed in this tab.

        The bridge from the **integrated predictive score (d)** and
        **contextual alignment score (c)** to the **priority score (p)** follows
        the selected contextual modulation operator (Linear, Geometric, or Minimum).
        """
    )

    ml_attributions = st.session_state.get("ml_attributions")
    fuzzy_details = st.session_state.get("fuzzy_details")
    master_data = st.session_state.get("master_data")
    context_config = st.session_state.get("context_config") or {}
    ml_details = st.session_state.get("ml_details")
    feature_config = st.session_state.get("feature_config") or {}
    aggregation_method = st.session_state.get("aggregation_method", "average")
    op = normalize_operator(modulation_operator)

    if ml_attributions is None or fuzzy_details is None or master_data is None:
        st.warning("Run the analysis first to enable case-level explanations.")
        return

    id_col = "CaseID"
    priority_col = _priority_column()
    case_options = df.sort_values(priority_col, ascending=False)[id_col].tolist()
    st.markdown(
        '<p style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.35rem;">'
        "Select case to explain"
        "</p>",
        unsafe_allow_html=True,
    )
    selected_case = st.selectbox(
        "Select case to explain",
        case_options,
        key="explain_case_select",
        label_visibility="collapsed",
    )

    case_idx = get_case_row_index(selected_case, df, id_col=id_col)
    if case_idx is None:
        st.error("Selected case was not found in the results table.")
        return

    case_row = df.loc[case_idx]
    master_row = master_data.loc[case_idx]
    fuzzy_row = fuzzy_details.loc[case_idx]

    _render_level4(
        selected_case,
        case_idx,
        case_row,
        master_row,
        ml_attributions,
        fuzzy_row,
        lambda_val,
        context_config,
        op,
    )
    st.divider()
    _render_level1(case_row, lambda_val, op)
    st.divider()
    _render_level2(case_idx, case_row, master_row, ml_attributions, ml_details, feature_config)
    st.divider()
    _render_level3(case_idx, fuzzy_row, aggregation_method, context_config)
