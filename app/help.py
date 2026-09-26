# help.py

def get_help_markdown() -> str:
    """
    Returns a concise Markdown help text explaining the CADEMAS-ML
    decision-making approach, required inputs, and an illustrative example.
    """
    return """

## What is CADEMAS-ML?

**CADEMAS-ML** is a *cooperative and context-aware decision support framework*.
It integrates:
- **Machine Learning risk estimates** (data-driven predictions), and
- **Contextual reasoning** based on expert-defined rules and organizational priorities.

Rather than replacing ML with rules (or vice versa), CADEMAS-ML **combines both explicitly and transparently** into a single prioritization score.

---

### Required Inputs (in order)

The application requires **four inputs**, all of which are mandatory to run an analysis.

---

#### 1. Model performance configuration (JSON)

This JSON defines, **for each ML model**, the performance metrics used to compute its contribution weight.

Typical contents:
- Model identifier (must match the uploaded MOJO filename),
- One or more performance metrics (e.g. AUC, accuracy, F1).

These metrics are **not used for prediction**, but **only to weight models** in the final ML risk aggregation.

> ⚠️ The identifiers in this file must be consistent with the names of the uploaded MOJO models.

##### Expected JSON structure (example)

```json
{
  "rf_model.zip": {
    "performance": {
      "AUC": 0.87,
      "Accuracy": 0.81,
      "F1": 0.78
    }
  },
  "gbm_model.zip": {
    "performance": {
      "AUC": 0.91,
      "Accuracy": 0.84,
      "F1": 0.80
    }
  }
}
```

Notes:
- Each top-level key **must exactly match** the filename of an uploaded MOJO model.
- Any number of performance metrics can be included.
- The metric used for weighting is selected **in the application UI**, not in this file.

---

#### 2. Context configuration (JSON)

This JSON defines the **contextual reasoning layer**, including:
- Atomic fuzzy rules (membership functions over features),
- Optional derived rules (logical combinations),
- Aggregation operators (AND / OR),
- Final context aggregation logic.

Each rule produces a **membership degree** μ in [0,1], and the full context evaluation yields a **context alignment score** C_i in [0,1].

##### Minimal illustrative context JSON (example)

```json
{
  "rules": [
    {
      "name": "AgeHigh",
      "feature": "Age",
      "type": "linear_increasing",
      "params": [40, 60]
    },
    {
      "name": "CriticalEquipment",
      "feature": "EquipmentType",
      "type": "categorical_map",
      "params": {
        "Critical": 1.0,
        "Standard": 0.5,
        "Low": 0.1
      }
    }
  ],
  "derived_rules": [
    {
      "name": "OperationalPriority",
      "op": "AND",
      "inputs": [
        { "rule": "AgeHigh" },
        { "rule": "CriticalEquipment" }
      ]
    }
  ],
  "logic": {
    "op": "AVERAGE",
    "inputs": [
      { "rule": "OperationalPriority" }
    ]
  }
}
```

Explanation:
- **rules** define atomic fuzzy membership functions over raw features.
- **derived_rules** combine atomic rules using fuzzy operators (`AND`, `OR`, `PRODUCT`, `AVERAGE`).
- **logic** defines a declarative final aggregation when the context engine is used without an explicit override.
- In the interactive application, the operator selected under **Context Parameters** takes precedence and is applied to the numeric atomic and derived memberships retained in the audit matrix.
- All membership values are normalized in the range `[0,1]`.

---

#### 3. Machine Learning models (MOJO files)

- Models must be uploaded in **H2O MOJO format** (`.zip`).
- Each uploaded MOJO must correspond to **one entry** in the performance JSON.
- The filename (or identifier) is used to:
  - retrieve its performance metrics,
  - compute its aggregation weight.

##### Important characteristics
- Models **may have been trained on different feature subsets**.
- There is **no requirement** that all models use the same input variables.
- Feature alignment is handled internally by H2O at prediction time.

---

#### 4. Dataset of cases (CSV)

This file contains the **cases to be evaluated** (e.g. employees, assets, clients).

Requirements:
- Must include all features required by **any** of the uploaded models or context rules.
- Features **do not need to be identical across models**.
- The dataset may contain *extra variables* not used by any model or rule.

- Optionally include a `Case_ID` or `ID` column to identify cases; otherwise the application assigns a consecutive `CaseID`. Identifier columns are excluded from model inference.
- CSV files may use **comma**, **semicolon**, or **tab** delimiters (auto-detected on upload).

---

## Included Example

The repository includes a ready-to-load example in [example_attrition/](https://github.com/pnovoa/cademas-app/tree/main/example_attrition).
It is intended as a practical reference for the input format rather than as domain documentation.

Load the files as follows:
- **Model Configuration (JSON)**: [model_definitions.json](https://github.com/pnovoa/cademas-app/blob/main/example_attrition/model_definitions.json)
- **Context Configuration (JSON)**: one or more files from [context/](https://github.com/pnovoa/cademas-app/tree/main/example_attrition/context), such as [context_crisis.json](https://github.com/pnovoa/cademas-app/blob/main/example_attrition/context/context_crisis.json) and [context_digital_transformation.json](https://github.com/pnovoa/cademas-app/blob/main/example_attrition/context/context_digital_transformation.json)
- **MOJO Models (.zip)**: the `.zip` model files in [models/](https://github.com/pnovoa/cademas-app/tree/main/example_attrition/models)
- **Dataset (.csv)**: [cases_atttrition.csv](https://github.com/pnovoa/cademas-app/blob/main/example_attrition/data/cases_atttrition.csv)

If multiple context JSON files are uploaded, select the desired one in **Context Parameters → Context** before running the analysis.

---

## Machine Learning Risk Aggregation

For each case:
1. Every MOJO model produces a **risk probability** R_i in [0,1].
2. Model weights w_i are computed from the selected performance metric.
3. The global ML risk is computed as a weighted sum of individual model risks.

This allows higher-performing models to contribute more strongly to the final risk estimate.

---

## Context Evaluation (Fuzzy Reasoning)

- Atomic rules map raw features to fuzzy membership values.
- Logical operators are implemented using standard fuzzy logic:
  - **OR** → `max`,
  - **AND** → `min` or `product` (depending on configuration).
- Conditional rules (e.g. department-specific criteria) are activated selectively.

The result is a **context alignment score** C_i, fully interpretable and auditable.

---

## Decision Integration

The final priority score is a convex combination of the global ML risk and the context alignment score, controlled by the parameter λ.

where:
- λ = 1 → purely ML-driven decision,
- λ = 0 → purely context-driven decision,
- Intermediate values allow **sensitivity analysis**.

---

## Explain Tab (Case-Level XAI)

After running an analysis, the **Explain** tab provides a case-level breakdown of the prioritization score without re-running model inference.

The tab includes four levels:

1. **Global decomposition** — shows how `P = λ·Ri + (1−λ)·Ci` combines ML risk and context alignment.
2. **ML risk attribution** — top features that push the cooperative ML risk up or down for the selected case. Attributions are precomputed during the analysis run using a one-at-a-time perturbation method (each feature replaced by a cohort baseline: median for numeric variables, mode for categorical variables).
3. **Fuzzy context traceability** — membership degrees of atomic and derived rules for the selected case. Under a strict minimum aggregation, the bottleneck rule is highlighted.
4. **Natural-language summary** — concise template-based explanation combining ML drivers and contextual constraints.

> Note: the bundled AutoML MOJO models are predominantly Stacked Ensembles, which do not expose native H2O SHAP contributions. The perturbation-based attributions are therefore approximate local effects rather than exact TreeSHAP values.

---

## Why This Approach?

CADEMAS-ML supports:
- Explicit trade-offs between prediction and policy,
- Robust decisions under changing organizational contexts,
- Full transparency: every score can be decomposed into ML and context components.

Use the **Robustness** tab to explore how rankings evolve as λ changes.

---
"""
