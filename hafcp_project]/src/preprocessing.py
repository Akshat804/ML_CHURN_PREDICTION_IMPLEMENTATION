import pandas as pd

def load_data(path):
    import pandas as pd
    df = pd.read_csv(path)

    # drop customer ID
    df = df.drop("customerID", axis=1)

    # convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # handle TotalCharges (string issue)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    # fill missing
    df = df.fillna(df.median(numeric_only=True))

    return df

def split_data(df):
    from sklearn.model_selection import train_test_split
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # Encode categorical
    X = pd.get_dummies(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)