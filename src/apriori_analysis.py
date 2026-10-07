"""Apriori and association-rule wrappers with predictable empty outputs."""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(basket: pd.DataFrame, min_support: float) -> pd.DataFrame:
    if basket.empty or basket.shape[1] == 0:
        return pd.DataFrame(columns=["support", "itemsets"])
    return apriori(basket, min_support=min_support, use_colnames=True, max_len=5, low_memory=True).sort_values("support", ascending=False).reset_index(drop=True)

def generate_rules(frequent_itemsets: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    if frequent_itemsets.empty or not frequent_itemsets["itemsets"].map(len).gt(1).any():
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    try:
        return association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    except (ValueError, KeyError):
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
