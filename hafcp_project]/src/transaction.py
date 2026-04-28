def create_transactions(fuzzy_df):
    cols = fuzzy_df.columns
    data = fuzzy_df.values

    transactions = []

    for row in data:
        transaction = []
        for i in range(len(cols)):
            val = str(row[i])

            # 🔥 skip nan
            if val == "nan":
                continue

            transaction.append(f"{cols[i]}_{val}")

        transaction = list(set(transaction))  # remove duplicates
        transactions.append(transaction)

    return transactions