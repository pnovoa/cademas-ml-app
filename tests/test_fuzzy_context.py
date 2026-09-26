import pandas as pd

from fuzzy_context import calculate_context_score


def test_final_logic_aggregation_overrides_linguistic_operator():
    data = pd.DataFrame({"x": [10.0], "y": [0.0]})
    context = {
        "rules": [
            {
                "name": "x_high",
                "feature": "x",
                "type": "linear_increasing",
                "params": [0, 10],
            },
            {
                "name": "y_high",
                "feature": "y",
                "type": "linear_increasing",
                "params": [0, 10],
            },
        ],
        "logic": {
            "op": "AND",
            "aggregation": "average",
            "inputs": [{"rule": "x_high"}, {"rule": "y_high"}],
        },
    }

    scores, _ = calculate_context_score(data, context, aggregation=None)

    assert scores.iloc[0] == 0.5
