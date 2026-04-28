import shap
import numpy as np

def compute_shap_importance(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    importance = np.mean(abs(shap_values), axis=0)
    
    return dict(zip(X_train.columns, importance))