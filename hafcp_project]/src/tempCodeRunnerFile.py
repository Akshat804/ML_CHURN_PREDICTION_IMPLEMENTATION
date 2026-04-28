
    X_train = X_train.sample(200, random_state=42)
    y_train = y_train.loc[X_train.index]
