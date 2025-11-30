#backend/comparision_utils.py
import json
import os
from typing import Dict, Any, List

NORMAL_PATH = "backend/normal_ranges.json"

if os.path.exists(NORMAL_PATH):
    with open(NORMAL_PATH, "r") as f:
        NORMALS = json.load(f)
else:
    NORMALS = {}


def build_comparisons(
    disease: str, inputs: Dict[str, Any]
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    comparisons: list of {feature, user_value, normal_min, normal_max, status}
    chart_data: list of {feature, user_value, normal_value}
    """
    disease_normals = NORMALS.get(disease, {})
    comparisons = []
    chart_data = []

    for feat, val in inputs.items():
        try:
            v = float(val)
        except Exception:
            continue

        norm = disease_normals.get(feat)
        if norm:
            n_min = norm.get("min")
            n_max = norm.get("max")
            status = "normal"
            if n_min is not None and v < n_min:
                status = "low"
            if n_max is not None and v > n_max:
                status = "high"

            comparisons.append(
                {
                    "feature": feat,
                    "user_value": v,
                    "normal_min": n_min,
                    "normal_max": n_max,
                    "status": status,
                }
            )

            normal_mid = (n_min + n_max) / 2 if n_min and n_max else n_max or n_min
            chart_data.append(
                {
                    "feature": feat,
                    "user_value": v,
                    "normal_value": normal_mid,
                }
            )

    return comparisons, chart_data
