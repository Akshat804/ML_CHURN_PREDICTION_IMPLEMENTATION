def compute_transaction_utility(transaction, feature_importance):
    utility = 0
    
    for item in transaction:
        feature = item.split("_")[0]
        utility += feature_importance.get(feature, 0)
    
    return utility