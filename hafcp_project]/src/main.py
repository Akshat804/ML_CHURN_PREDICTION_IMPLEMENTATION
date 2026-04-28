from preprocessing import load_data, split_data
from model import train_model
from shap_utils import compute_shap_importance
from fuzzy import apply_fuzzy
from transaction import create_transactions
from chud import top_k_hui
from hafcp_feature import create_hafcp_feature

import numpy as np


def main():
    #  Step 1: Load Data
    print("Step 1: Loading data...")
    df = load_data("../data/telcome_churn.csv")

    #  Step 2: Split Data
    print("Step 2: Splitting...")
    X_train, X_test, y_train, y_test = split_data(df)

    #  OPTIONAL sampling
    X_train = X_train.sample(200, random_state=42)
    y_train = y_train.loc[X_train.index]

    #  Step 3: Train Model
    print("Step 3: Training model...")
    model = train_model(X_train, y_train)

    #  Step 4: Feature Importance
    print("Step 4: SHAP importance...")
    feature_importance = compute_shap_importance(model, X_train)

    #  Select top features
    top_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:10]
    X_train = X_train[top_features]

    #  Align importance
    feature_importance = {k: v for k, v in feature_importance.items() if k in top_features}

    #  Reduce dominance
    if "Age" in feature_importance:
        feature_importance["Age"] *= 0.7

    #  Compute overall churn EARLY (fix bug)
    overall_churn = np.mean(y_train)

    #  Step 5: Fuzzy transform (FULL DATA for validation later)
    print("Step 5: Fuzzy transform...")
    fuzzy_full = apply_fuzzy(X_train)
    transactions_full = create_transactions(fuzzy_full)

    # FILTER churn-only for mining
    X_train_churn = X_train[y_train == 1]
    fuzzy_churn = apply_fuzzy(X_train_churn)
    transactions_churn = create_transactions(fuzzy_churn)

    # Step 7: Run CHUD on churn-only
    print("Step 7: Running CHUD...")
    patterns = top_k_hui(transactions_churn, feature_importance, k=10)

    #  Step 8: Filter VALID patterns
    print("\n Top VALID HAFCP Patterns:")

    valid_patterns = []

    for p, u in patterns:
        #  evaluate on FULL dataset (correct)
        feat = create_hafcp_feature(transactions_full, p)

        indices = [i for i, val in enumerate(feat) if val == 1]

        if len(indices) == 0:
            continue

        churn_rate = np.mean(y_train.iloc[indices])
        support = len(indices) / len(feat)

        #  filter meaningful patterns
        if churn_rate > overall_churn:
            valid_patterns.append((p, u, churn_rate, support))

    # 🔹 print results
    for i, (p, u, cr, sup) in enumerate(valid_patterns[:5]):
        print(f"Rank {i+1}: {p} | Utility: {round(u,4)} | Churn: {round(cr,4)} | Support: {round(sup,4)}")

    # 🔹 Step 9: Pick best valid pattern
    if len(valid_patterns) > 0:
       best = max(valid_patterns, key=lambda x: x[2])  # x[2] = churn rate
       top_pattern = best[0]
    else:
        print("\n No strong churn patterns found")
        return

    print("\n Top Pattern Selected:", top_pattern)

    # 🔹 Step 10: Final evaluation
    hafcp_feat = create_hafcp_feature(transactions_full, top_pattern)

    indices = [i for i, val in enumerate(hafcp_feat) if val == 1]

    support = len(indices) / len(hafcp_feat)
    churn_rate_pattern = np.mean(y_train.iloc[indices]) if len(indices) > 0 else 0

    print("\nSupport:", round(support, 4))
    print(" Churn rate (pattern):", round(churn_rate_pattern, 4))
    print(" Overall churn rate:", round(overall_churn, 4))

    print("\n Done!")


if __name__ == "__main__":
    main()