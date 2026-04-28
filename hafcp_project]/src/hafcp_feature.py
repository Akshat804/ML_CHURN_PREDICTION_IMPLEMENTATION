def create_hafcp_feature(transactions, pattern):
    pattern_set = set(pattern)
    
    feature = []
    
    for t in transactions:
        feature.append(int(pattern_set.issubset(set(t))))
    
    return feature