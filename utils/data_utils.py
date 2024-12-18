# utils/data_utils.py

import pandas as pd
from typing import Tuple, Dict


def process_data_pipeline_a(data: str) -> Tuple[pd.DataFrame, Dict]:
    # Deterministic processing logic for Pipeline A
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    description_dict = {"summary": "Pipeline A processed the data.", "columns": df.columns.tolist()}
    return df, description_dict


def process_data_pipeline_b(data: str) -> Tuple[pd.DataFrame, Dict]:
    # Deterministic processing logic for Pipeline B
    df = pd.DataFrame({"X": [7, 8, 9], "Y": [10, 11, 12]})
    description_dict = {"summary": "Pipeline B processed the data.", "columns": df.columns.tolist()}
    return df, description_dict


def analyze_full_data(dataframe_dict: Dict) -> Dict:
    # Convert dict back to DataFrame
    df = pd.DataFrame.from_dict(dataframe_dict)
    # Deterministic final analysis logic
    overview_info = {"mean_values": df.mean().to_dict(), "total_entries": len(df)}
    return overview_info
