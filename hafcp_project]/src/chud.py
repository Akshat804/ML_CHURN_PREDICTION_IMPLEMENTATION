from itertools import combinations

def top_k_hui(transactions, feature_importance, k=5):

    patterns = {}
    support_count = {}

    n = len(transactions)


    for t in transactions:
        t = list(set(t))   

        max_len = min(3, len(t))

        for l in range(1, max_len + 1):
            for subset in combinations(t, l):
                subset = tuple(sorted(subset))

                
                util = sum(
                    feature_importance.get(item.split("_")[0], 0)
                    for item in subset
                )

                
                util *= len(subset)

                
                patterns[subset] = patterns.get(subset, 0) + util

                
                support_count[subset] = support_count.get(subset, 0) + 1

    
    for p in patterns:
        support = support_count[p] / n


        patterns[p] = patterns[p] * support

    
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)

    return sorted_patterns[:k]