import numpy as np
import pandas as pd

def fuzzy_transform_column(col):
    low = np.percentile(col, 33)
    high = np.percentile(col, 66)
    
    labels = []
    for val in col:
        if val <= low:
            labels.append("L")
        elif val <= high:
            labels.append("M")
        else:
            labels.append("H")
    return labels

def apply_fuzzy(X):
    fuzzy_df = pd.DataFrame()

    for col in X.columns:
        col_data = X[col]

        # 🔥 remove NaN BEFORE processing
        col_data = col_data.fillna(col_data.mode()[0] if col_data.dtype == 'object' else col_data.median())

        if col_data.dtype != 'uint8' and col_data.nunique() > 10:
            fuzzy_df[col] = fuzzy_transform_column(col_data)
        else:
            fuzzy_df[col] = col_data.astype(str)

    return fuzzy_df