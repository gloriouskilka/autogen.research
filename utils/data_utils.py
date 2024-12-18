import pandas as pd
from typing import Tuple, Dict, Any


def process_data_pipeline_a(data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict]:
    # Deterministic processing logic for Pipeline A
    overview = {
        "columns": data.columns.tolist(),
        "num_rows": len(data),
        "summary_statistics": data.describe().to_dict(),
    }
    return data, overview


def process_data_pipeline_b(data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict]:
    # Deterministic processing logic for Pipeline B
    data["new_column"] = data.select_dtypes(include="number").mean(axis=1)
    overview = {
        "columns": data.columns.tolist(),
        "num_rows": len(data),
        "added_new_column": True,
    }
    return data, overview


def analyze_full_data(dataframe: pd.DataFrame) -> Dict:
    # Deterministic final analysis logic
    correlation = dataframe.corr().to_dict()
    overview_info = {
        "correlation_matrix": correlation,
        "num_rows": len(dataframe),
    }
    return overview_info
