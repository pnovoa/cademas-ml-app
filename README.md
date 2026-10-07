# CADEMAS-ML

Software tool for **cooperative and context-aware prioritization**. It operationalizes a configurable class of [CADEMAS](https://doi.org/10.1109/cai68641.2026.11536392) systems in which automated decision-making (ADM) units are machine-learning models and contextual knowledge is encoded as declarative fuzzy rules.

CADEMAS-ML integrates and weights heterogeneous predictive models (H2O MOJO), evaluates fuzzy context specifications (JSON), combines predictive and contextual scores with configurable modulation operators (Linear, Geometric, Minimum) and an adjustable trade-off parameter \(\lambda\), and supports case-level explainability and ranking robustness analysis.

**Keywords:** automated decision-making; fuzzy decision making; context-aware decision support; machine learning; prioritization; cooperative systems

## Main views

| View | Role |
|------|------|
| **Home** | Upload inputs, choose metric / context / aggregation / modulation, run analysis |
| **Overview** | Priority scores \(p\), integrated predictive scores \(d\), contextual alignment \(c\) |
| **Models** | Model weights and per-ADM predictions |
| **Context** | Membership functions, derived rules, alignment audit |
| **Explain** | Natural-language summary, local perturbation attributions, fuzzy traceability |
| **Robustness** | Rank trajectories, rank distributions, and rank acceptability under alternative \(\lambda\) / operators |

## Run locally

Python 3.11 and Java 17+ (required for H2O MOJO loading; Streamlit Cloud installs OpenJDK 21 via `packages.txt`). On macOS: `brew install openjdk@17` or `openjdk@21`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app_v1.py
```

For Streamlit Community Cloud, set the main file to `app/app_v1.py`.

## Example inputs

A ready-to-load employee-attrition bundle is in `example_attrition/`:

- Model configuration: `example_attrition/models/model_definitions.json`
- Context configurations: `example_attrition/context/*.json` (*Economic Crisis* and *Digital / Technological Transformation*)
- MOJO models: `example_attrition/models/*.zip`
- Dataset: `example_attrition/data/cases_atttrition.csv`

## Architecture diagram

Editable draw.io source: `stuffs/instantiation-architecture.drawio`.

## License

MIT. Source: [https://github.com/pnovoa/cademas-app](https://github.com/pnovoa/cademas-app).
