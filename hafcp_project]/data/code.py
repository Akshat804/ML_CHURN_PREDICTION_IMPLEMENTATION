import kagglehub

# Download latest version
path = kagglehub.dataset_download("subhasishsinha/bank-customer-churn-modelling")

print("Path to dataset files:", path)