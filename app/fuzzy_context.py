import numpy as np
import pandas as pd

def get_membership(x, m_type, params):
    """Router that selects the correct membership function."""
    if m_type == 'triangular':
        return triangular(x, *params)
    elif m_type == 'trapezoidal':
        return trapezoidal(x, *params)
    elif m_type == 'linear_increasing':
        return linear_increasing(x, *params)
    elif m_type == 'linear_decreasing':
        return linear_decreasing(x, *params)
    else:
        return np.zeros_like(x, dtype=float)

def get_categorical_membership(series: pd.Series, m_type: str, params):
    """Categorical membership for rule types:
    - categorical_map: params is a dict {category: score}
    - categorical_set: params is a dict of sets, and scores are assigned via a map in derived rules
    """
    if m_type == "categorical_map":
        mapping = params if isinstance(params, dict) else {}
        return series.map(mapping).fillna(0.0).astype(float).values

    if m_type == "categorical_set":
        # categorical_set itself returns the set name (string) per row, not a numeric score.
        # We encode as object array; derived rules can map set->score.
        groups = params if isinstance(params, dict) else {}
        out = np.array([None] * len(series), dtype=object)
        for set_name, categories in groups.items():
            if not isinstance(categories, list):
                continue
            mask = series.isin(categories)
            out[mask.values] = set_name
        # Unmatched categories remain None
        return out

    return np.zeros(len(series), dtype=float)

def apply_when_mask(df: pd.DataFrame, when: dict) -> np.ndarray:
    """Build a boolean mask for a simple conditional rule application.

    Supported forms:
      when = {"feature": "Department", "equals": "Sales"}
    """
    if not when or not isinstance(when, dict):
        return np.ones(len(df), dtype=bool)

    feat = when.get("feature")
    if feat is None or feat not in df.columns:
        return np.zeros(len(df), dtype=bool)

    if "equals" in when:
        return (df[feat] == when["equals"]).values

    if "in" in when and isinstance(when["in"], list):
        return df[feat].isin(when["in"]).values

    return np.ones(len(df), dtype=bool)

def triangular(x, a, b, c):
    """Pico en b."""
    if a == b or b == c: return np.zeros_like(x)
    term1 = (x - a) / (b - a)
    term2 = (c - x) / (c - b)
    return np.maximum(0, np.minimum(term1, term2))

def trapezoidal(x, a, b, c, d):
    """Plano (1) entre b y c."""
    # max(min( (x-a)/(b-a), 1, (d-x)/(d-c) ), 0)
    term1 = (x - a) / (b - a) if b > a else np.ones_like(x) # Protección div/0
    term3 = (d - x) / (d - c) if d > c else np.ones_like(x)
    return np.maximum(0, np.minimum(np.minimum(term1, 1), term3))

def linear_increasing(x, a, b):
    """Gamma (Sube): 0 hasta a, sube, 1 desde b en adelante."""
    if a == b: return (x >= a).astype(float)
    y = (x - a) / (b - a)
    return np.maximum(0, np.minimum(y, 1))

def linear_decreasing(x, a, b):
    """L-Shape (Baja): 1 hasta a, baja, 0 desde b en adelante."""
    if a == b: return (x <= a).astype(float)
    y = (b - x) / (b - a)
    return np.maximum(0, np.minimum(y, 1))


def _agg_values(op: str, arr: np.ndarray) -> float:
    op = (op or "AVERAGE").upper()
    if arr.size == 0:
        return 0.0
    if op in ("AVERAGE", "MEAN"):
        return float(np.mean(arr))
    if op in ("MIN", "MINIMUM"):
        return float(np.min(arr))
    if op in ("MAX",):
        return float(np.max(arr))
    if op in ("PRODUCT", "PROD"):
        return float(np.prod(arr))
    if op in ("AND",):
        # Fuzzy AND (Gödel t-norm)
        return float(np.min(arr))
    if op in ("OR",):
        # Fuzzy OR (Gödel s-norm)
        return float(np.max(arr))
    return float(np.mean(arr))

def _eval_logic_node(node, mu_dict: dict, n_rows: int) -> np.ndarray:
    """Evaluate a logic node into a vector of size n_rows.

    Node forms supported:
      - {"rule": "rule_name"}
      - {"op": "AND"|"OR"|"AVERAGE"|"PRODUCT"|"MIN"|"MAX", "inputs": [ ... ]}

    Returns a numeric array in [0,1] (best effort).
    """
    if node is None:
        return np.zeros(n_rows, dtype=float)

    # Direct reference to a rule
    if isinstance(node, dict) and "rule" in node:
        key = node["rule"]
        val = mu_dict.get(key)
        if val is None:
            return np.zeros(n_rows, dtype=float)
        # categorical_set returns object array; cannot be used directly without mapping
        if val.dtype == object:
            return np.zeros(n_rows, dtype=float)
        return val.astype(float)

    # Composite node
    if isinstance(node, dict) and "op" in node and "inputs" in node:
        # A final logic node may declare an aggregation semantics separately
        # from its linguistic connective (e.g., AND with average aggregation).
        op = (node.get("aggregation") or node.get("op") or "AVERAGE").upper()
        inputs = node.get("inputs") or []
        if not isinstance(inputs, list) or len(inputs) == 0:
            return np.zeros(n_rows, dtype=float)

        mats = []
        for child in inputs:
            mats.append(_eval_logic_node(child, mu_dict, n_rows))
        mat = np.vstack(mats)  # shape: (k, n_rows)

        if op in ("AVERAGE", "MEAN"):
            return np.mean(mat, axis=0)
        if op in ("MIN", "MINIMUM", "AND"):
            return np.min(mat, axis=0)
        if op in ("MAX", "OR"):
            return np.max(mat, axis=0)
        if op in ("PRODUCT", "PROD"):
            return np.prod(mat, axis=0)

        return np.mean(mat, axis=0)

    return np.zeros(n_rows, dtype=float)

def calculate_context_score(df: pd.DataFrame, context_config: dict, aggregation: str):
    """Compute contextual alignment score (c) supporting:

    - context_config['rules'] : base membership rules
    - context_config['derived_rules'] : composition rules built from base rules
    - context_config['logic'] : final logic expression for Ci

    Backward compatible with the old schema (rules only).

    Returns:
      scores: pd.Series
      fuzzy_details: pd.DataFrame (all computed mu_* columns + derived)
    """

    rules = context_config.get("rules", []) if isinstance(context_config, dict) else []
    derived_rules = context_config.get("derived_rules", []) if isinstance(context_config, dict) else []
    logic = context_config.get("logic") if isinstance(context_config, dict) else None

    n = len(df)

    # Store all rule outputs here by name
    mu = {}

    # Details dataframe for audit
    fuzzy_scores = pd.DataFrame(index=df.index)

    # --- 1) Base rules ---
    for i, rule in enumerate(rules):
        col = rule.get('feature')
        m_type = rule.get('type')
        params = rule.get('params')
        rule_name = rule.get('name') or f"rule_{i}_{col}"  # alias
        when = rule.get("when")

        # Default outputs
        out = np.zeros(n, dtype=float)

        # Conditional mask
        mask = apply_when_mask(df, when)

        # Numeric rule
        if col in df.columns and m_type in ("triangular", "trapezoidal", "linear_increasing", "linear_decreasing"):
            x = df[col].astype(float).values
            tmp = get_membership(x, m_type, params)
            out[mask] = tmp[mask]
            mu[rule_name] = out
            fuzzy_scores[f"mu_{rule_name}"] = out
            continue

        # Categorical rule
        if col in df.columns and m_type in ("categorical_map", "categorical_set"):
            series = df[col].astype(str)
            tmp = get_categorical_membership(series, m_type, params)
            # categorical_set returns object array (set names)
            if isinstance(tmp, np.ndarray) and tmp.dtype == object:
                # Apply mask by setting non-matching rows to None
                tmp2 = tmp.copy()
                tmp2[~mask] = None
                mu[rule_name] = tmp2
                fuzzy_scores[f"mu_{rule_name}"] = tmp2
            else:
                out[mask] = tmp[mask]
                mu[rule_name] = out
                fuzzy_scores[f"mu_{rule_name}"] = out
            continue

        # Missing feature or unsupported type
        mu[rule_name] = out
        fuzzy_scores[f"mu_{rule_name}"] = out

    # --- 2) Derived rules ---
    for d in derived_rules:
        d_name = d.get("name")
        op = (d.get("op") or "AVERAGE").upper()
        inputs = d.get("inputs") or []
        if not d_name or not isinstance(inputs, list) or len(inputs) == 0:
            continue

        # Evaluate row-wise
        out = np.zeros(n, dtype=float)

        for row_idx in range(n):
            vals = []
            for inp in inputs:
                if not isinstance(inp, dict) or "rule" not in inp:
                    continue
                key = inp["rule"]
                base_val = mu.get(key)
                if base_val is None:
                    continue

                # categorical_set mapping support: {"rule": "jobrole_critical", "map": {"critical": 1.0, ...}}
                if isinstance(base_val, np.ndarray) and base_val.dtype == object:
                    mapper = inp.get("map") if isinstance(inp.get("map"), dict) else {}
                    set_name = base_val[row_idx]
                    vals.append(float(mapper.get(set_name, 0.0)))
                else:
                    vals.append(float(base_val[row_idx]))

            if len(vals) == 0:
                out[row_idx] = 0.0
            else:
                out[row_idx] = _agg_values(op, np.array(vals, dtype=float))

        mu[d_name] = out
        fuzzy_scores[f"mu_{d_name}"] = out

    # --- 3) Final context score ---
    # Priority: explicit aggregation > logic (default) > safe mean

    numeric_cols = fuzzy_scores.select_dtypes(include=[np.number])

    scores = None

    # 1) Explicit aggregation if provided
    if aggregation == "average":
        scores = numeric_cols.mean(axis=1)
    elif aggregation == "minimum (strict)":
        scores = numeric_cols.min(axis=1)
    elif aggregation == "product":
        scores = numeric_cols.prod(axis=1)

    # 2) Default: logic expression (if present)
    if scores is None and logic and isinstance(logic, dict):
        scores_arr = _eval_logic_node(logic, mu, n)
        scores = pd.Series(scores_arr, index=df.index)

    # 3) Final fallback: mean of numeric memberships
    if scores is None:
        scores = numeric_cols.mean(axis=1)

    # Clip to [0,1] for safety
    scores = scores.clip(0.0, 1.0)

    return scores, fuzzy_scores




if __name__ == "__main__":

    # -----------------------------
    # 1) Dataset de prueba
    # -----------------------------
    df = pd.DataFrame({
        "salary": [30, 50, 80],
        "tenure": [1, 5, 10],
        "department": ["Sales", "IT", "HR"]
    })

    # -----------------------------
    # 2) Configuración del contexto
    # -----------------------------
    context_config = {
        "rules": [
            {
                "name": "low_salary",
                "feature": "salary",
                "type": "linear_decreasing",
                "params": [40, 70]
            },
            {
                "name": "high_tenure",
                "feature": "tenure",
                "type": "linear_increasing",
                "params": [3, 8]
            },
            {
                "name": "dept_group",
                "feature": "department",
                "type": "categorical_set",
                "params": {
                    "critical": ["IT"],
                    "non_critical": ["Sales", "HR"]
                }
            }
        ],

        "derived_rules": [
            {
                # Producto (AND fuerte)
                "name": "economic_pressure",
                "op": "PRODUCT",
                "inputs": [
                    {"rule": "low_salary"},
                    {"rule": "high_tenure"}
                ]
            },
            {
                # OR semántico sobre departamentos
                "name": "dept_importance",
                "op": "MAX",
                "inputs": [
                    {
                        "rule": "dept_group",
                        "map": {
                            "critical": 1.0,
                            "non_critical": 0.3
                        }
                    }
                ]
            }
        ],

        # -----------------------------
        # 3) Lógica final (NO min global)
        # -----------------------------
        "logic": {
            "op": "AVERAGE",
            "inputs": [
                {"rule": "economic_pressure"},
                {"rule": "dept_importance"}
            ]
        }
    }

    # -----------------------------
    # 4) Cálculo
    # -----------------------------
    scores, details = calculate_context_score(
        df,
        context_config,
        aggregation="minimum (strict)"  # <-- se ignora porque hay logic
    )

    # -----------------------------
    # 5) Resultados
    # -----------------------------
    print("\n=== DATA ===")
    print(df)

    print("\n=== MEMBERSHIPS & DERIVED ===")
    print(details.round(3))

    print("\n=== FINAL CONTEXT SCORE ===")
    print(scores.round(3))