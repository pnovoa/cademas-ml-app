from traceback import print_last

import base64
import io
import streamlit as st
import h2o
import pandas as pd
import numpy as np
import json
import os
import tempfile
import altair as alt
from animation import render_animated_header
from h2o.model import ModelBase
from fuzzy_context import calculate_context_score, get_membership
from explain_utils import aggregate_weighted_attributions, humanize_rule
from ml_attribution import compute_model_attributions
from modulation import (
    USES_LAMBDA,
    additive_contributions,
    build_q_mod,
    default_top_n_config,
    find_config_by_label,
    modulate,
    normalize_operator,
    operator_display_label,
    operator_formula_caption,
    uses_lambda,
)
from view_explanations import render_explain_tab

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "assets", "cademas.png")
LOGO_SVG_PATH = os.path.join(APP_DIR, "assets", "cademas_logo.svg")
MINISTRY_LOGO_PATH = os.path.join(APP_DIR, "assets", "logo_ministerio.jpg")

CASE_ID_COL = "CaseID"
CASE_ID_SOURCE_COLUMNS = ("Case_ID", "ID")
CSV_DELIMITERS = (",", ";", "\t")


def _risk_progress_column(label: str) -> st.column_config.ProgressColumn:
    return st.column_config.ProgressColumn(
        label,
        format="percent",
        min_value=0,
        max_value=1,
        color="auto-inverse",
    )


def _context_progress_column(label: str) -> st.column_config.ProgressColumn:
    return st.column_config.ProgressColumn(
        label,
        format="%.2f",
        min_value=0,
        max_value=1,
        color="auto-inverse",
    )


def read_uploaded_csv(file) -> pd.DataFrame:
    file.seek(0)
    text = file.read().decode("utf-8-sig")
    file.seek(0)

    best_df = None
    max_cols = 0

    for sep in CSV_DELIMITERS:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
        except (pd.errors.ParserError, ValueError):
            continue
        if len(df.columns) > max_cols and len(df) > 0:
            max_cols = len(df.columns)
            best_df = df

    if best_df is None:
        raise ValueError(
            "Could not parse the dataset. Supported delimiters: comma, semicolon, and tab."
        )

    return best_df


def prepare_dataset_case_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for source_col in CASE_ID_SOURCE_COLUMNS:
        if source_col in df.columns:
            df.insert(0, CASE_ID_COL, df[source_col].values)
            return df.drop(columns=[source_col])

    df.insert(0, CASE_ID_COL, np.arange(1, len(df) + 1))
    return df


def model_features_df(df: pd.DataFrame, id_col: str = CASE_ID_COL) -> pd.DataFrame:
    return df.drop(columns=[id_col])


def image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    extension = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime_subtype = "jpeg" if extension == "jpg" else extension
    return f"data:image/{mime_subtype};base64,{encoded_image}"

st.set_page_config(
    page_title="CADEMAS-ML – Cooperative and Context-Aware Decision Support",
    page_icon=LOGO_PATH,
    layout="wide",
)

custom_css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:11pt;
    align-items:center;
    }
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: anywhere;
        line-height: 1.3;
        max-width: 100%;
    }
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: anywhere;
        line-height: 1.3;
    }
    [data-testid="stMetric"] {
        align-items: flex-start;
        min-height: 5.5rem;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# --- 1. Inicialización ---
@st.cache_resource
def init_h2o():
    try:
        h2o.init(max_mem_size="700m", nthreads=1)
        return None
    except Exception as e:
        return str(e)


# --- 2. Funciones Lógicas ---
def save_temp_file(uploaded_file):
    try:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    except:
        return None


def load_uploaded_json(uploaded_file):
    uploaded_file.seek(0)
    return json.load(uploaded_file)


def render_home_tab(results_ready=False):
    st.markdown(
        """
        Welcome to **CADEMAS-ML**, a cooperative and context-aware decision support
        system <a href="#ref-1">[1]</a>, <a href="#ref-2">[2]</a>.
        """,
        unsafe_allow_html=True,
    )

    anim_col, text_col = st.columns([1.35, 1], gap="large")
    with anim_col:
        render_animated_header()
    with text_col:
        st.markdown(
            """
            ### How it works

            CADEMAS-ML combines machine learning predictions with expert-defined
            context rules to prioritize cases in a transparent and auditable way.

            1. **Upload four inputs** in the sidebar: model configuration (JSON),
               context configuration (JSON), MOJO models (.zip), and the case dataset (CSV).
            2. **Run the analysis** to compute, for each case:
               - **Integrated predictive score (d)** — weighted combination of model predictions
               - **Contextual alignment score (c)** — fuzzy evaluation of contextual rules
            3. **Choose a contextual modulation operator** (Linear, Geometric, or Minimum)
               to combine d and c into a **priority score (p)**. For Linear and
               Geometric, adjust **λ** to balance the predictive and contextual signals.
            4. **Explore the results** in Overview, Models, Context, **Explain**, and
               **Robustness** (within-operator λ sweeps and across-operator comparisons).
            """
        )

    st.divider()
    st.markdown(
        """
        <p id="ref-1">
        <strong>[1]</strong> Novoa-Hernández, P., Pelta, D. A., Godz, M.,
        Verdegay, J. L., &amp; Buendia-Carrillo, D. (2026).
        CADEMAS – A Framework for Cooperative Automated Decision-Making Systems.
        In <em>2026 IEEE Conference on Artificial Intelligence (CAI)</em>
        (pp. 756–761). IEEE.
        <a href="https://doi.org/10.1109/cai68641.2026.11536392" target="_blank" rel="noopener noreferrer">
        https://doi.org/10.1109/cai68641.2026.11536392</a>
        </p>

        <p id="ref-2">
        <strong>[2]</strong> Godz, M., Novoa-Hernández, P., &amp; Pelta, D. A.
        (2026). A Cooperative and Context-Aware Approach for Employee Attrition
        Prevention. In <em>Communications in Computer and Information Science</em>
        (pp. 203–217). Springer Nature Switzerland.
        <a href="https://doi.org/10.1007/978-3-032-29000-7_15" target="_blank" rel="noopener noreferrer">
        https://doi.org/10.1007/978-3-032-29000-7_15</a>
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_help_tab():
    from help import get_help_markdown

    st.markdown(get_help_markdown())


def render_about_tab():
   # st.subheader("About")

    st.markdown(
        """
        **Authors:**
        - **Pavel Novoa Hernández** (Universidad de La Laguna, Spain) — pnovoahe@ull.edu.es
        - David A. Pelta (Universidad de Granada, Spain) — dpelta@ugr.es
        
        **Funding:**
        
        This application has been funded by the project:
        *Study, Analysis and Evaluation of Cooperative Automated Decision-Making Systems (CADEMAS)* \
        (reference number **PID2023-146575NB-I00**), funded by **MCIU/AEI/10.13039/501100011033** and by **FSE+**.
        [More information](https://modougr.es/index.php/proyecto_cademas/)
        """
    )
    ministry_logo_data_uri = image_to_data_uri(MINISTRY_LOGO_PATH)
    st.markdown(
        f"""
        <a href="https://www.ciencia.gob.es" target="_blank" rel="noopener noreferrer">
            <img
                src="{ministry_logo_data_uri}"
                width="512"
                alt="Ministerio de Ciencia, Innovación y Universidades"
            >
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_app_title():
    logo_col, title_col = st.columns([0.08, 0.92], vertical_alignment="center")
    with logo_col:
        st.image(LOGO_SVG_PATH, width=64)
    with title_col:
        st.title("CADEMAS-ML")


# --- 3. SIDEBAR ---
feature_config = None
selected_metric = None
context_configs = []
selected_context_config = None
aggregation_method = st.session_state.get("aggregation_method", "average")
lambda_val = st.session_state.get("lambda_val", 0.5)
run_calc = False

with st.sidebar:
    if st.session_state.get("base_results") is not None:
        st.success(
            "Analysis results are ready. Use the result tabs to inspect the outputs."
        )

    header_col, upload_hint_col = st.columns([0.9, 0.1], vertical_alignment="center")
    with header_col:
        st.title("Configuration")
    with upload_hint_col:
        upload_hint = st.empty()

    with st.expander("1. Files and Data", expanded=True):
        json_ml = st.file_uploader("Model Configuration (JSON)", type=['json'])
        json_context_files = st.file_uploader(
            "Context Configuration (JSON)",
            type=['json'],
            accept_multiple_files=True
        )
        model_files = st.file_uploader("MOJO Models (.zip)", type=['zip'], accept_multiple_files=True)
        data_file = st.file_uploader(
            "Dataset (.csv)",
            type=['csv'],
            help=(
                "Cases are identified by a 'Case_ID' or 'ID' column when present; "
                "otherwise a consecutive 'CaseID' is assigned automatically. "
                "Identifier columns are excluded from model inference. "
                "CSV files may use comma, semicolon, or tab delimiters."
            ),
        )
        st.session_state.record_id_col = CASE_ID_COL

    files_ready = bool(json_ml and json_context_files and model_files and data_file)
    if not files_ready:
        upload_hint.markdown(
            """
            <div style="text-align: right;">
                <span
                    title="Upload required files to unlock settings."
                    style="cursor: help; color: #64748b; font-size: 1rem;"
                >&#9432;</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if files_ready:
        with st.expander("2. ML Parameters", expanded=False):
            feature_config = load_uploaded_json(json_ml)
            first_key = list(feature_config.keys())[0]
            metrics = list(feature_config[first_key].get("performance", {}).keys())
            if "selected_metric" not in st.session_state:
                st.session_state.selected_metric = metrics[0] if metrics else None
            st.selectbox(
                "Metric used for model weighting (w)",
                metrics,
                key="selected_metric"
            )
            selected_metric = st.session_state.selected_metric

        with st.expander("3. Context Parameters", expanded=False):
            for context_file in json_context_files:
                try:
                    context_config = load_uploaded_json(context_file)
                    context_name = context_config.get("context_name") or context_file.name
                    context_configs.append({
                        "label": context_name,
                        "config": context_config,
                    })
                except Exception as e:
                    st.error(f"Could not load context file '{context_file.name}': {e}")

            if context_configs:
                if st.session_state.get("selected_context_idx", 0) not in range(len(context_configs)):
                    st.session_state.selected_context_idx = 0
                selected_context_idx = st.selectbox(
                    "Context",
                    range(len(context_configs)),
                    format_func=lambda i: context_configs[i]["label"],
                    key="selected_context_idx"
                )
                selected_context_config = context_configs[selected_context_idx]["config"]
                selected_context_description = selected_context_config.get("description")
                if selected_context_description:
                    st.caption(selected_context_description)
            else:
                st.info("Upload one or more context configuration JSON files.")

            if "aggregation_method" not in st.session_state:
                st.session_state.aggregation_method = "average"
            st.selectbox(
                "Context aggregation",
                ["average", "minimum (strict)", "product"],
                key="aggregation_method",
                help=(
                    "How atomic and derived fuzzy memberships are combined into "
                    "the contextual alignment score (c). Distinct from the contextual "
                    "modulation operator that combines d and c into the priority score (p)."
                ),
            )
            aggregation_method = st.session_state.aggregation_method

            if "modulation_operator" not in st.session_state:
                st.session_state.modulation_operator = "Linear"
            st.selectbox(
                "Contextual modulation operator",
                ["Linear", "Geometric", "Minimum"],
                key="modulation_operator",
            )
            st.caption(operator_formula_caption(st.session_state.modulation_operator))

        run_calc = st.button("Run analysis", type="primary", width='stretch')

        st.markdown("## Decision Adjustment")
        modulation_operator = normalize_operator(
            st.session_state.get("modulation_operator", "Linear")
        )
        if uses_lambda(modulation_operator):
            lambda_val = st.slider(
                "Lambda (λ)",
                0.0,
                1.0,
                0.5,
                0.01,
                key="lambda_val",
            )
            if modulation_operator == "linear":
                st.caption(
                    f"Contextual weight: {1 - lambda_val:.0%} | "
                    f"Predictive weight: {lambda_val:.0%}"
                )
            else:
                st.caption(
                    f"Contextual exponent: {1 - lambda_val:.2f} | "
                    f"Predictive exponent: {lambda_val:.2f}"
                )
        else:
            lambda_val = float(st.session_state.get("lambda_val", 0.5))
            st.caption(
                "Minimum is parameter-free: p = min{d, c}. "
                "λ is not used for this operator."
            )


# --- 4. ESTADO ---
if 'base_results' not in st.session_state: st.session_state.base_results = None
if 'fuzzy_details' not in st.session_state: st.session_state.fuzzy_details = None
if 'ml_details' not in st.session_state: st.session_state.ml_details = None
if 'context_config' not in st.session_state: st.session_state.context_config = None  # Guardamos config para graficar
if 'master_data' not in st.session_state: st.session_state.master_data = None  # Guardamos raw data para histogramas
if 'p_label' not in st.session_state: st.session_state.p_label = None
if 'ml_attributions' not in st.session_state: st.session_state.ml_attributions = None
if 'feature_config' not in st.session_state: st.session_state.feature_config = None


# --- 5. EJECUCIÓN ---
if run_calc:
    st.session_state["run_triggered"] = True
if st.session_state.get("run_triggered", False):
    if feature_config and model_files and data_file and selected_metric and selected_context_config:
        with st.spinner(text="Processing...", show_time=True):
            try:
                h2o_error = init_h2o()
                if h2o_error:
                    st.session_state["run_triggered"] = False
                    st.error(
                        "H2O could not be initialized. The app interface is available, "
                        "but MOJO model inference requires an H2O runtime."
                    )
                    st.exception(RuntimeError(h2o_error))
                    st.stop()

                # A. Carga
                data_file.seek(0)
                master_df = prepare_dataset_case_ids(read_uploaded_csv(data_file))
                st.session_state.master_data = master_df
                model_df = model_features_df(master_df)
                context_config = selected_context_config
                st.session_state.context_config = context_config  # Guardar config

                # B. Pesos ML
                valid_models = [m.name for m in model_files if m.name in feature_config]


                metrics_vals = {m: feature_config[m]["performance"].get(selected_metric, 0) for m in valid_models}



                total = sum(metrics_vals.values())
                weights = {m: (val / total if total > 0 else 1 / len(valid_models)) for m, val in metrics_vals.items()}

                st.session_state.ml_details = {
                    "weights": weights,
                    "metric": selected_metric,
                    "performance": metrics_vals,
                }
                st.session_state.feature_config = feature_config

                # C. H2O Loop
                risk_accum = np.zeros(len(master_df))
                temp_results = master_df.copy()
                by_model_attribs = {}
                attribution_warnings = []

                prog_bar = st.progress(0)
                valid_model_files = [m for m in model_files if m.name in weights]
                total_model_steps = max(len(valid_model_files), 1)
                p_col = None

                for i, m_file in enumerate(valid_model_files):
                    path = save_temp_file(m_file)
                    hf = None
                    try:
                        mojo = h2o.import_mojo(path)
                        hf = h2o.H2OFrame(model_df)
                        preds = mojo.predict(hf).as_data_frame()
                        p_col = 'p1' if 'p1' in preds.columns else preds.columns[-1]
                        current_p_label = preds.columns[-1]
                        st.session_state.p_label = current_p_label if st.session_state.p_label is None else st.session_state.p_label
                        if st.session_state.p_label != current_p_label:
                            st.error(f"Error: Predicted label is not the same across models.")
                            st.stop()
                        vals = preds[p_col].values
                        risk_accum += vals * weights[m_file.name]
                        temp_results[f"{m_file.name.split('.')[0]}_prob"] = vals

                        model_features = feature_config.get(m_file.name, {}).get("features", [])
                        contrib_df, warns = compute_model_attributions(
                            mojo,
                            model_df,
                            model_features,
                            p_col_hint=p_col,
                        )
                        by_model_attribs[m_file.name] = contrib_df
                        attribution_warnings.extend(warns)
                    finally:
                        if os.path.exists(path): os.remove(path)
                        if hf is not None:
                            h2o.remove(hf)
                    prog_bar.progress((i + 1) / total_model_steps)

                st.session_state.ml_attributions = {
                    "by_model": by_model_attribs,
                    "aggregated": aggregate_weighted_attributions(by_model_attribs, weights),
                    "meta": {
                        "method": "perturbation_one_at_a_time",
                        "baseline": "cohort_median_or_mode",
                        "features_by_model": {
                            name: feature_config.get(name, {}).get("features", [])
                            for name in by_model_attribs
                        },
                        "warnings": attribution_warnings,
                    },
                }

                temp_results["Ri_Global_Risk"] = risk_accum

                # D. Contexto
                ci_scores, fuzzy_df = calculate_context_score(master_df, context_config, aggregation_method)
                temp_results["Ci_Context_Score"] = ci_scores


                st.session_state.base_results = temp_results
                st.session_state.fuzzy_details = fuzzy_df

                st.session_state["run_triggered"] = False
                st.success("Computation completed successfully.")

            except Exception as e:
                st.session_state["run_triggered"] = False
                st.error(f"Error: {e}")
    else:
        st.error("Required input files are missing.")

# --- 6. VISUALIZACIÓN ---

render_app_title()

if st.session_state.base_results is not None:
    df = st.session_state.base_results.copy()
    modulation_operator = normalize_operator(
        st.session_state.get("modulation_operator", "Linear")
    )
    df["Prioritization_Score "] = modulate(
        df["Ri_Global_Risk"],
        df["Ci_Context_Score"],
        modulation_operator,
        lambda_val,
    )
    ml_contrib, ctx_contrib = additive_contributions(
        df["Ri_Global_Risk"],
        df["Ci_Context_Score"],
        modulation_operator,
        lambda_val,
    )
    if ml_contrib is not None:
        df["ML_Contribution"] = ml_contrib
        df["Context_Contribution"] = ctx_contrib
    else:
        df["ML_Contribution"] = np.nan
        df["Context_Contribution"] = np.nan
    df["Lambda"] = lambda_val
    df["Modulation_Operator"] = operator_display_label(modulation_operator)

    home_tab, tab1, tab2, tab3, tab_explain, tab_robustness, tab_help, tab4 = st.tabs(
        ["Home", "Overview", "Models", "Context", "Explain", "Robustness", "Help", "About"]
    )

    with home_tab:
        render_home_tab(results_ready=True)

    # --- TAB 1 ---
    with tab1:
        overview_metrics = [
            ("Positive label", f"{st.session_state.p_label}"),
            ("Number of cases", f"{len(df['Prioritization_Score '])}"),
            ("Average Priority Score", f"{df['Prioritization_Score '].mean():.1%}"),
            ("Average Integrated Predictive Score (d)", f"{df['Ri_Global_Risk'].mean():.1%}"),
            ("Average Contextual Alignment Score (c)", f"{df['Ci_Context_Score'].mean():.1%}"),
            (
                "High Priority Cases (> 0.75)",
                len(df[df["Prioritization_Score "] > 0.75]),
            ),
        ]
        metric_row_1 = st.columns(3)
        metric_row_2 = st.columns(3)
        for col, (label, value) in zip([*metric_row_1, *metric_row_2], overview_metrics):
            with col:
                st.metric(label=label, value=value)

        st.divider()

        g1, g2 = st.columns([1.5, 1])
        with g1:
            st.subheader("Priority: predictive vs contextual")
            # Scatter Plot con Altair
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Ri_Global_Risk', title='Integrated predictive score d (0–1)'),
                y=alt.Y('Ci_Context_Score', title='Contextual alignment score c (0–1)'),
                color=alt.Color('Prioritization_Score ', scale=alt.Scale(scheme='turbo'), title='Priority score'),
                tooltip=['CaseID', 'Ri_Global_Risk', 'Ci_Context_Score', 'Prioritization_Score ']
            ).interactive()
            st.altair_chart(scatter, width='stretch')

        with g2:
            st.subheader("Priority score")
            # Histograma Mejorado con Altair
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("Prioritization_Score ", bin=alt.Bin(step=0.1), title="Priority score range"),
                y=alt.Y('count()', title='Number of Cases'),
                color=alt.value("#1E88E5")
            )
            st.altair_chart(hist, width='stretch')

        st.subheader("Prioritized Cases")

        # 1. Definimos las columnas a mostrar
        id_col = "CaseID"
        prob_cols = [c for c in df.columns if c.endswith("_prob")]
        # Columnas calculadas (en orden lógico)
        score_cols = [
            "Prioritization_Score ",
            "Ri_Global_Risk",
            "Ci_Context_Score",
            "Modulation_Operator",
        ]
        if uses_lambda(modulation_operator):
            score_cols.append("Lambda")
        if modulation_operator == "linear":
            score_cols.extend(["ML_Contribution", "Context_Contribution"])
        calculated_cols = [id_col] + score_cols + prob_cols

        # Columnas originales (preservando su orden)
        original_cols = [c for c in st.session_state.master_data.columns if c not in calculated_cols]

        # Orden final: calculadas primero, luego dataset original completo
        final_cols = calculated_cols + original_cols

        # 2. Aplicamos Estilos con Pandas (Heatmap)
        cols_main = ["Prioritization_Score ", "Ri_Global_Risk", "Ci_Context_Score"]

        styled_df = df[final_cols].sort_values("Prioritization_Score ", ascending=False).style \
            .format("{:.1%}", subset=cols_main) \
            .background_gradient(cmap='RdYlGn_r', subset=['Prioritization_Score '], vmin=0, vmax=1)
        # Nota: RdYlGn_r pone Verde en 0 (Bajo riesgo) y Rojo en 1 (Alto riesgo)

        # 3. Renderizamos con Column Config para toques extra
        st.dataframe(
            styled_df,
            column_config={
                "Prioritization_Score ": st.column_config.NumberColumn(
                    "Priority score (p)",
                    help="Priority score from contextual modulation of d and c",
                    format="percent",
                ),
                "Ri_Global_Risk": _risk_progress_column("Integrated predictive score (d)"),
                "Ci_Context_Score": _context_progress_column("Contextual alignment score (c)"),
                "Modulation_Operator": st.column_config.TextColumn(
                    "Modulation operator",
                    help="Contextual modulation operator used to compute the priority score",
                ),
            },
            width='stretch',
            height=500
        )

    # --- TAB 2 ---
    with tab2:
        st.markdown("""
        This view presents the **Decision Integration Layer**: how uploaded models
        contribute predictive signals that are combined into a single
        **integrated predictive score (d)** before contextual modulation.

        Two sections are included:

        - **Model Weights** — contribution weight ($w_k$) of each model, derived from the
          performance metric selected in the sidebar.
        - **Model predictions** — per-case outputs of each model, together with the
          weighted **integrated predictive score (d)**.
        """)

        st.subheader("Model Weights")
        if st.session_state.ml_details:
            weights = st.session_state.ml_details["weights"]
            metric_name = st.session_state.ml_details.get("metric") or "Performance"
            performance = st.session_state.ml_details.get("performance") or {}
            if not performance and st.session_state.get("feature_config"):
                feature_config = st.session_state.feature_config
                performance = {
                    model_name: feature_config.get(model_name, {})
                    .get("performance", {})
                    .get(metric_name, np.nan)
                    for model_name in weights
                }
            w_df = pd.DataFrame(
                [
                    {
                        "Model": model_name,
                        metric_name: performance.get(model_name, np.nan),
                        "Weight (Wi)": weight,
                    }
                    for model_name, weight in weights.items()
                ]
            )
            st.dataframe(
                w_df,
                width="stretch",
                column_config={
                    metric_name: st.column_config.NumberColumn(
                        metric_name,
                        help="Selected performance metric used to derive Weight (Wi).",
                        format="%.4f",
                    ),
                    "Weight (Wi)": st.column_config.NumberColumn(
                        "Weight (Wi)",
                        help=f"Normalized weight from {metric_name}.",
                        format="%.4f",
                    ),
                },
            )
        st.subheader("Model predictions")
        id_col = "CaseID"
        prob_cols = [c for c in df.columns if c.endswith("_prob")]

        # Construimos tabla extendida de probabilidades + riesgo global
        weights = st.session_state.ml_details["weights"]

        prob_df = df[[id_col] + prob_cols].copy()

        # Calcular Riesgo Global explícito (suma ponderada)
        global_risk = np.zeros(len(prob_df))
        for col in prob_cols:
            model_name = col.replace("_prob", "")
            # Buscamos el peso correspondiente (por nombre de modelo)
            for w_key, w_val in weights.items():
                if w_key.startswith(model_name):
                    global_risk += prob_df[col].values * w_val
                    break

        prob_df["Ri_Global_Risk"] = global_risk

        prob_column_config = {
            id_col: st.column_config.TextColumn(id_col),
            "Ri_Global_Risk": _risk_progress_column("Integrated predictive score (d)"),
        }
        for col in prob_cols:
            model_label = col.replace("_prob", "").replace("_", " ")
            prob_column_config[col] = _risk_progress_column(model_label)

        st.dataframe(
            prob_df,
            column_config=prob_column_config,
            width='stretch'
        )

    # --- TAB 3 (Visualización Difusa con Altair) ---
    with tab3:
        st.markdown("""
        This view presents the **Contextual Modulator** inputs: how expert-defined
        fuzzy rules are applied to each case and aggregated into the
        **contextual alignment score (c)** used together with the integrated
        predictive score before modulation.

        Three sections are included:

        - **Membership Functions** — inspect how a selected atomic rule maps input values to
          membership degrees ($\\mu$), overlaid on the case distribution. A **Numerical Audit**
          table alongside the chart reports per-case raw feature values and computed membership
          degrees for the selected rule.
        - **Derived Rules Overview** — membership values of composite rules and the resulting
          **contextual alignment score** across all cases.
        - **Scatterplot of Derived Rules** — explore the relationship between two derived
          rules, colored by contextual alignment (when applicable).
        """)

        st.subheader("Membership Functions")

        # 1. Selector de Regla Inteligente
        rules = st.session_state.context_config['rules']
        context_config = st.session_state.context_config

        # Diccionario para nombres bonitos en la UI
        type_labels = {
            'triangular': 'Triangle',
            'trapezoidal': 'Trapezoidal',
            'linear_increasing': 'Linear increasing',
            'linear_decreasing': 'Linear decreasing'
        }

        # Generamos la lista de opciones formateadas
        rule_options = []
        for i, r in enumerate(rules):
            r_type = r.get('type')
            r_feat = r.get('feature')
            r_name = r.get('name')

            if r_type in ('categorical_map', 'categorical_set'):
                clean_type = 'Categorical'
            else:
                clean_type = type_labels.get(r_type, r_type)

            label = humanize_rule(r_name, context_config) if r_name else r_feat
            rule_options.append(f"{label} [{clean_type}]")

        selected_rule_idx = st.selectbox(
            "Select variable / rule to inspect:",
            range(len(rules)),
            format_func=lambda i: rule_options[i]
        )

        # Recuperamos la regla seleccionada
        selected_rule = rules[selected_rule_idx]
        feat = selected_rule.get('feature')
        m_type = selected_rule.get('type')
        params = selected_rule.get('params')
        rule_alias = selected_rule.get('name') or feat
        rule_label = humanize_rule(rule_alias, context_config)

        # 2. Verificar datos y preparar visualización
        raw_data = st.session_state.master_data
        col_chart, col_audit = st.columns([1.4, 1], gap="medium")

        with col_chart:
            if feat in raw_data.columns and m_type not in ('categorical_map', 'categorical_set'):

                # --- Lógica de Renderizado Altair ---

                # A. Preparar Rango X
                data_vals = raw_data[feat].dropna()
                min_d, max_d = data_vals.min(), data_vals.max()

                # El rango del gráfico debe cubrir los datos Y los parámetros de la regla
                # Concatenamos params con min/max de los datos para encontrar los límites
                all_points = list(params) + [min_d, max_d]
                x_start = min(all_points) * 0.95
                x_end = max(all_points) * 1.05

                x_grid = np.linspace(x_start, x_end, 300)

                # B. Calcular Y (Membresía) usando la función genérica
                y_grid = get_membership(x_grid, m_type, params)

                line_df = pd.DataFrame({'x_val': x_grid, 'membership': y_grid})

                # C. Preparar Líneas Verticales de Referencia
                labels = []
                if m_type == 'triangular':
                    labels = ['a (Inicio)', 'b (Pico)', 'c (Fin)']
                elif m_type == 'trapezoidal':
                    labels = ['a (Inicio)', 'b (Plano in)', 'c (Plano out)', 'd (Fin)']
                elif m_type == 'linear_increasing':
                    labels = ['a (Base 0)', 'b (Tope 1)']
                elif m_type == 'linear_decreasing':
                    labels = ['a (Tope 1)', 'b (Base 0)']
                else:
                    labels = [f"p{i}" for i in range(len(params))]

                rules_df = pd.DataFrame({
                    'x_pos': params,
                    'label': labels,
                    'color': ['red'] * len(params)
                })

                # --- D. Construcción del Gráfico (Altair) ---

                # Capa 1: Histograma
                hist = alt.Chart(raw_data).mark_bar(color='#e0e0e0', opacity=0.7).encode(
                    x=alt.X(feat, bin=alt.Bin(maxbins=30), title=feat),
                    y=alt.Y('count()', title='Frequency'),
                    tooltip=['count()']
                )

                # Capa 2: Curva Membresía
                line = alt.Chart(line_df).mark_line(color="#FF4B4B", strokeWidth=3).encode(
                    x='x_val',
                    y=alt.Y('membership', title='Membership (μ)', scale=alt.Scale(domain=[0.0, 1])),
                    tooltip=[alt.Tooltip('x_val', format='.2f'), alt.Tooltip('membership', format='.2f')]
                )

                # Capa 3: Referencias Verticales
                refs = alt.Chart(rules_df).mark_rule(strokeDash=[5, 5],  strokeWidth=2, color='black', opacity=0.5).encode(
                    x='x_pos',
                    tooltip=['label', 'x_pos']
                )

                # Combinar con ejes independientes
                final_chart = alt.layer(hist, line, refs).resolve_scale(
                    y='independent'
                ).properties(
                    height=350,
                    title=f"Membership and Frequency for '{rule_alias}'"
                )

                st.altair_chart(final_chart, width='stretch')

            elif feat in raw_data.columns and m_type in ('categorical_map', 'categorical_set'):
                st.info(f"Categorical rule '{rule_label}' selected. Membership curves are only available for numeric rules.")
            else:
                st.warning(f"Column '{feat}' is not present in the dataset.")

        with col_audit:
            st.markdown(
                "<p style='font-size: 14px; font-weight: bold; margin: 0 0 0.5rem 0;'>"
                "Numerical Audit</p>",
                unsafe_allow_html=True,
            )

            feature_name = f"mu_{rule_alias}"
            if feature_name in st.session_state.fuzzy_details.columns:
                id_col = "CaseID"

                audit_df = pd.concat([
                    raw_data[[id_col, feat]].reset_index(drop=True),
                    st.session_state.fuzzy_details[[feature_name]].reset_index(drop=True)
                ], axis=1)

                if feat in raw_data.columns and m_type in ('categorical_map', 'categorical_set'):
                    st.dataframe(
                        audit_df,
                        width='stretch',
                        height=350,
                    )
                else:
                    st.dataframe(
                        audit_df,
                        column_config={
                            feature_name: _context_progress_column(f"μ ({rule_label})"),
                        },
                        width='stretch',
                        height=350,
                    )
            else:
                st.info("No membership values available for the selected rule.")

        # --- Derived Rules Visualization ---
        st.subheader("Derived Rules Overview")
        derived_rules = st.session_state.context_config.get('derived_rules', [])
        if derived_rules:
            derived_cols = [r['name'] for r in derived_rules]
            derived_mu_cols = [f"mu_{c}" for c in derived_cols if f"mu_{c}" in st.session_state.fuzzy_details.columns]

            if derived_mu_cols:
                heatmap_df = st.session_state.fuzzy_details[derived_mu_cols].copy()
                heatmap_df['Context_Alignment'] = st.session_state.base_results['Ci_Context_Score']
                heatmap_df.insert(
                    0,
                    "CaseID",
                    st.session_state.master_data["CaseID"].values
                )
                rename_map = {
                    col: humanize_rule(col[3:], context_config)
                    for col in derived_mu_cols
                }
                rename_map["Context_Alignment"] = "Contextual alignment score (c)"
                display_df = heatmap_df.rename(columns=rename_map)
                derived_column_config = {
                    "CaseID": st.column_config.TextColumn("CaseID"),
                    "Contextual alignment score (c)": _context_progress_column(
                        "Contextual alignment score (c)"
                    ),
                }
                for col in derived_mu_cols:
                    derived_column_config[rename_map[col]] = _context_progress_column(rename_map[col])

                st.dataframe(
                    display_df,
                    column_config=derived_column_config,
                    width='stretch',
                    height=400
                )

                # Optional: scatterplot of two selected derived rules
                if len(derived_mu_cols) >= 2:
                    st.subheader("Scatterplot of Derived Rules")
                    rule_label = lambda col: humanize_rule(col[3:], context_config)
                    col1, col2 = st.columns(2)
                    with col1:
                        x_rule = st.selectbox(
                            "X-axis derived rule",
                            derived_mu_cols,
                            index=0,
                            format_func=rule_label,
                        )
                    with col2:
                        y_rule = st.selectbox(
                            "Y-axis derived rule",
                            derived_mu_cols,
                            index=1,
                            format_func=rule_label,
                        )

                    if x_rule != y_rule:
                        scatter_df = st.session_state.fuzzy_details[[x_rule, y_rule]].copy()
                        scatter_df['CaseID'] = st.session_state.master_data['CaseID']
                        scatter_df['Context_Alignment'] = st.session_state.base_results['Ci_Context_Score']
                        scatter_plot = alt.Chart(scatter_df).mark_circle(size=60).encode(
                            x=alt.X(x_rule, title=rule_label(x_rule)),
                            y=alt.Y(y_rule, title=rule_label(y_rule)),
                            color=alt.Color('Context_Alignment', scale=alt.Scale(scheme='turbo')),
                            tooltip=['CaseID', x_rule, y_rule]
                        ).interactive()
                        st.altair_chart(scatter_plot, width='stretch')
                    else:
                        st.warning("Please select different derived rules.")
            else:
                st.info("Derived rules exist but no computed values are available.")
        else:
            st.info("No derived rules defined in the context configuration.")


    # --- TAB: EXPLAIN ---
    with tab_explain:
        render_explain_tab(df, lambda_val, modulation_operator)


    # --- TAB: ROBUSTNESS ---
    with tab_robustness:
        id_col = "CaseID"

        st.markdown("""
        This analysis evaluates how **stable case rankings** remain under alternative
        contextual modulation configurations $Q_{\\mathrm{mod}}$.

        - **Within-operator** — for Linear and Geometric, vary $\\lambda$ from pure
          contextual alignment ($\\lambda = 0$) to a pure integrated predictive signal
          ($\\lambda = 1$).
        - **Across-operator** — compare Linear, Geometric, and Minimum while holding
          the integrated predictive score (d) and contextual alignment score (c) fixed.

        Three complementary views are provided:

        - **Bump Chart** — how each case's priority rank shifts across selected configurations.
        - **Ranks Box Plot** — distribution of ranks per case over $Q_{\\mathrm{mod}}$.
        - **Rank Acceptability Indices** — relative frequency of each case occupying
          each rank position across configurations.

        Use **Settings** to choose which modulation operators to include, the $\\lambda$
        sweep granularity (shared by Linear and Geometric), how many cases to display,
        and which configuration ranks the Top N cases.
        """)

        st.subheader("Settings")
        global_op_label = operator_display_label(modulation_operator)

        # Persist checkbox defaults from the global sidebar operator on first visit /
        # when the global operator changes and the user has not customized yet.
        default_ops_key = "robustness_default_ops_source"
        if st.session_state.get(default_ops_key) != global_op_label:
            st.session_state.rob_op_linear = global_op_label == "Linear"
            st.session_state.rob_op_geometric = global_op_label == "Geometric"
            st.session_state.rob_op_minimum = global_op_label == "Minimum"
            st.session_state[default_ops_key] = global_op_label

        st.markdown("**Modulation operators in $Q_{\\mathrm{mod}}$**")
        op_c1, op_c2, op_c3 = st.columns(3)
        with op_c1:
            rob_linear = st.checkbox("Linear", key="rob_op_linear")
        with op_c2:
            rob_geometric = st.checkbox("Geometric", key="rob_op_geometric")
        with op_c3:
            rob_minimum = st.checkbox("Minimum", key="rob_op_minimum")

        selected_ops = []
        if rob_linear:
            selected_ops.append("linear")
        if rob_geometric:
            selected_ops.append("geometric")
        if rob_minimum:
            selected_ops.append("minimum")

        if not selected_ops:
            st.warning(
                "Select at least one modulation operator to build $Q_{\\mathrm{mod}}$."
            )
        else:
            needs_lambda_sweep = any(op in USES_LAMBDA for op in selected_ops)

            c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
            with c_ctrl1:
                if needs_lambda_sweep:
                    n_partitions = st.slider(
                        "Lambda Partitions (Steps)",
                        min_value=2,
                        max_value=10,
                        value=4,
                        help="Shared λ grid for Linear and Geometric.",
                    )
                else:
                    n_partitions = 4
                    st.caption("λ partitions not used (only Minimum selected).")
            with c_ctrl2:
                top_n_show = st.slider(
                    "Show Top N Cases",
                    min_value=5,
                    max_value=50,
                    value=15,
                    help="Filters the cases with the highest ranking under the selected configuration.",
                )

            lambda_steps = (
                np.linspace(0, 1, n_partitions + 1) if needs_lambda_sweep else np.array([0.5])
            )
            q_mod = build_q_mod(selected_ops, lambda_steps)
            config_labels = [c.label for c in q_mod]
            preferred = default_top_n_config(q_mod, modulation_operator, preferred_lambda=0.5)
            preferred_label = preferred.label if preferred is not None else config_labels[0]

            with c_ctrl3:
                # Keep selection valid when Q_mod changes.
                current_sel = st.session_state.get("rob_top_n_config")
                if current_sel not in config_labels:
                    st.session_state.rob_top_n_config = preferred_label
                top_n_config_label = st.selectbox(
                    "Select Top N by configuration",
                    config_labels,
                    key="rob_top_n_config",
                    help=(
                        "Cases shown in the charts are the Top N under this modulation "
                        "configuration. Default prefers the sidebar operator at λ ≈ 0.5 "
                        "(or Minimum)."
                    ),
                )

            top_n_config = find_config_by_label(q_mod, top_n_config_label) or preferred

            # DATA PREPARATION over Q_mod
            bump_data = []
            for cfg_idx, cfg in enumerate(q_mod):
                temp_df = df[[id_col, "Ri_Global_Risk", "Ci_Context_Score"]].copy()
                lam = 0.5 if cfg.lambda_val is None else float(cfg.lambda_val)
                temp_df["Sim_Score"] = modulate(
                    temp_df["Ri_Global_Risk"],
                    temp_df["Ci_Context_Score"],
                    cfg.operator,
                    lam,
                )
                temp_df["Rank"] = temp_df["Sim_Score"].rank(ascending=False, method="first")
                temp_df["Operator"] = cfg.operator_label
                temp_df["Lambda"] = cfg.lambda_val if cfg.lambda_val is not None else np.nan
                temp_df["Config"] = cfg.label
                temp_df["ConfigOrder"] = cfg_idx
                bump_data.append(temp_df)

            bump_df = pd.concat(bump_data, ignore_index=True)

            # Top N by selected configuration ranks
            select_ranks = (
                bump_df[bump_df["Config"] == top_n_config.label]
                .set_index(id_col)["Rank"]
                .sort_values()
            )
            top_ids = select_ranks.head(top_n_show).index.tolist()
            filtered_bump_df = bump_df[bump_df[id_col].isin(top_ids)].copy()
            last_config = config_labels[-1]
            filtered_bump_df["IsLastConfig"] = filtered_bump_df["Config"] == last_config

            rank_stats = filtered_bump_df.groupby(id_col)["Rank"].agg(
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
            rank_stats["iqr"] = rank_stats["q3"] - rank_stats["q1"]
            rank_order = (
                rank_stats.sort_values(["median", "iqr"], ascending=[True, True])
                .index.tolist()
            )

            decision_grid_color = "#D3D3D3"

            case_color = alt.Color(
                f"{id_col}:N",
                legend=None,
                scale=alt.Scale(domain=rank_order),
            )

            case_labels = filtered_bump_df[id_col].astype(str).unique()
            max_label_len = max((len(label) for label in case_labels), default=1)
            # Right padding for end labels; keep the plot flush to the Y axis on the left.
            chart_right_padding = int(min(320, max(90, max_label_len * 10 + 48)))
            label_dx = int(max(10, min(28, 6 + max_label_len * 0.8)))
            n_configs = len(config_labels)
            # Quantitative axis: hug Y axis on the left; leave room on the right for labels.
            x_domain = [-0.02, max(n_configs - 1, 0) + 0.55]
            escaped_labels = [
                lbl.replace("\\", "\\\\").replace("'", "\\'") for lbl in config_labels
            ]
            label_expr = (
                "[" + ", ".join(f"'{lbl}'" for lbl in escaped_labels) + "][datum.value]"
            )

            base = alt.Chart(filtered_bump_df).encode(
                x=alt.X(
                    "ConfigOrder:Q",
                    title="Modulation configuration",
                    scale=alt.Scale(domain=x_domain, nice=False, zero=False),
                    axis=alt.Axis(
                        values=list(range(n_configs)),
                        labelExpr=label_expr,
                        labelAngle=-35,
                        labelLimit=180,
                        labelOverlap=False,
                    ),
                ),
                y=alt.Y(
                    "Rank:Q",
                    title="Ranking (1 = Highest Priority)",
                    scale=alt.Scale(
                        reverse=True,
                        zero=False,
                        domain=[0.5, filtered_bump_df["Rank"].max() + 0.5],
                    ),
                    axis=alt.Axis(tickMinStep=1),
                ),
                color=case_color,
            )

            lines = base.mark_line(
                interpolate="monotone", strokeWidth=4, clip=False
            ).encode(
                tooltip=[
                    alt.Tooltip(id_col, title="ID"),
                    alt.Tooltip("Config", title="Configuration"),
                    alt.Tooltip("Operator", title="Operator"),
                    alt.Tooltip("Lambda", format=".2f"),
                    alt.Tooltip("Rank", title="Ranking"),
                    alt.Tooltip("Sim_Score", title="Score", format=".1%"),
                ]
            )

            points = base.mark_circle(size=130, opacity=1, clip=False).encode(
                tooltip=[
                    alt.Tooltip(id_col),
                    alt.Tooltip("Config"),
                    alt.Tooltip("Rank"),
                ]
            )

            text_end = (
                base.mark_text(
                    align="left",
                    baseline="middle",
                    dx=label_dx,
                    fontSize=12,
                    clip=False,
                )
                .encode(text=f"{id_col}:N")
                .transform_filter(alt.datum.IsLastConfig)
            )

            final_chart = (
                alt.layer(lines, points, text_end)
                .configure_view(strokeWidth=0, clip=False)
                .configure(
                    padding={
                        "left": 2,
                        "right": chart_right_padding,
                        "top": 12,
                        "bottom": 12,
                    }
                )
            )


            st.altair_chart(final_chart, width="stretch", theme="streamlit", height=520)

            box_chart = (
                alt.Chart(filtered_bump_df)
                .mark_boxplot(
                    extent="min-max",
                    box=alt.MarkConfig(stroke="black"),
                    median=alt.MarkConfig(stroke="black"),
                )
                .encode(
                    x=alt.X(
                        f"{id_col}:N",
                        title="Case",
                        sort=rank_order,
                        scale=alt.Scale(paddingInner=0.2, paddingOuter=0.05),
                    ),
                    y=alt.Y(
                        "Rank:Q",
                        title="Ranking (1 = Highest Priority)",
                        scale=alt.Scale(reverse=True, zero=False),
                        axis=alt.Axis(
                            tickMinStep=1,
                            grid=True,
                            gridColor=decision_grid_color,
                        ),
                    ),
                    color=case_color,
                    tooltip=[id_col, alt.Tooltip("Rank", title="Rank")],
                )
            )

            n_configs = len(q_mod)
            rai_df = (
                filtered_bump_df.groupby([id_col, "Rank"])
                .size()
                .reset_index(name="count")
            )
            rai_df["frequency"] = rai_df["count"] / n_configs
            rai_df["freq_label"] = rai_df["frequency"].map(lambda x: f"{x:.1f}")

            # RdYlGn_r-like stops (green → red); avoids a matplotlib dependency on Cloud
            rai_color_range = [
                "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf",
                "#fee08b", "#fdae61", "#f46d43", "#d73027",
            ]

            rai_axis = alt.Axis(
                grid=True,
                gridColor="black",
                gridDash=[1, 3],
                tickBand="extent",
            )
            rai_rect = (
                alt.Chart(rai_df)
                .mark_rect()
                .encode(
                    x=alt.X(
                        f"{id_col}:O",
                        title="Case",
                        sort=rank_order,
                        axis=rai_axis,
                    ),
                    y=alt.Y(
                        "Rank:O",
                        title="Rank",
                        sort=alt.SortOrder("ascending"),
                        axis=rai_axis,
                    ),
                    color=alt.Color(
                        "frequency:Q",
                        title="Relative Frequency",
                        scale=alt.Scale(domain=[0, 1], range=rai_color_range),
                        legend=alt.Legend(
                            orient="top",
                            direction="horizontal",
                            titleOrient="top",
                            gradientLength=300,
                        ),
                    ),
                    tooltip=[
                        id_col,
                        alt.Tooltip("Rank", title="Rank"),
                        alt.Tooltip("frequency", title="Relative Frequency", format=".1%"),
                        alt.Tooltip("count", title="Occurrences"),
                    ],
                )
            )
            rai_text = (
                alt.Chart(rai_df)
                .mark_text(color="black", fontSize=10)
                .encode(
                    x=alt.X(f"{id_col}:O", sort=rank_order),
                    y=alt.Y("Rank:O", sort=alt.SortOrder("ascending")),
                    text="freq_label:N",
                )
            )
            rai_chart = alt.layer(rai_rect, rai_text).properties(
                height=max(200, rai_df["Rank"].nunique() * 25)
            )

            n_cases_shown = len(top_ids)
            box_caption = (
                "Distribution of priority rankings per case across selected modulation configurations."
            )
            rai_caption = (
                "Relative frequency of each case occupying a given rank across $Q_{\\mathrm{mod}}$."
            )

            if n_cases_shown <= 10:
                col_box, col_rai = st.columns(2)
                with col_box:
                    st.subheader("Ranks Box Plot")
                    st.caption(box_caption)
                    st.altair_chart(box_chart, width="stretch", theme="streamlit", height=400)
                with col_rai:
                    st.subheader("Rank Acceptability Indices")
                    st.caption(rai_caption)
                    st.altair_chart(rai_chart, width="stretch", theme="streamlit")
            else:
                st.subheader("Ranks Box Plot")
                st.caption(box_caption)
                st.altair_chart(box_chart, width="stretch", theme="streamlit", height=400)

                st.subheader("Rank Acceptability Indices")
                st.caption(rai_caption)
                st.altair_chart(rai_chart, width="stretch", theme="streamlit")

# --- TAB HELP ---

    with tab_help:
        render_help_tab()


# --- TAB 4 ---
    with tab4:
        render_about_tab()



else:
    home_tab, tab_help, tab4 = st.tabs(["Home", "Help", "About"])

    with home_tab:
        render_home_tab()

    with tab_help:
        render_help_tab()

    with tab4:
        render_about_tab()

