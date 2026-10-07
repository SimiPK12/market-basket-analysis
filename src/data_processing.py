"""Read and normalize common basket-analysis CSV formats."""
from io import BytesIO
import re
import pandas as pd

def load_data(file_bytes: bytes, filename: str = "uploaded.csv") -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty or len(df.columns) == 0:
        raise ValueError("The CSV has no data rows or columns.")
    return df

def _column_by_name(columns, candidates):
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    return next((lookup[key] for key in candidates if key in lookup), None)

def preprocess_transactions(df: pd.DataFrame):
    if df is None or df.empty:
        raise ValueError("The dataset contains no transactions.")
    tx_col = _column_by_name(df.columns, ["transactionid", "transaction", "basketid", "orderid", "invoiceid", "invoiceno", "invoice"])
    item_col = _column_by_name(df.columns, ["item", "product", "productname", "description"])
    rows_missing = int(df.isna().any(axis=1).sum())
    if tx_col is not None and item_col is not None and tx_col != item_col:
        clean = df[[tx_col, item_col]].dropna().copy()
        clean[tx_col] = clean[tx_col].astype(str).str.strip()
        clean[item_col] = clean[item_col].astype(str).str.strip()
        clean = clean[(clean[tx_col] != "") & (clean[item_col] != "")]
        # Retail invoice exports can include cancellations/returns rather than purchases.
        normalized_tx = re.sub(r"[^a-z0-9]", "", str(tx_col).lower())
        if normalized_tx in {"invoiceno", "invoice"}:
            clean = clean[~clean[tx_col].str.upper().str.startswith("C")]
            quantity_col = _column_by_name(df.columns, ["quantity", "qty"])
            if quantity_col is not None:
                positive_quantity = pd.to_numeric(df.loc[clean.index, quantity_col], errors="coerce") > 0
                clean = clean.loc[positive_quantity]
        grouped = clean.groupby(tx_col, sort=False)[item_col].apply(lambda x: sorted(set(x)))
        source_format = f"Long format ({tx_col}, {item_col})"
    else:
        # Wide transaction rows: split string cells on commas/semicolons/pipes; non-empty cells are products.
        possible = [c for c in df.columns if re.sub(r"[^a-z0-9]", "", str(c).lower()) not in {"transactionid", "transaction", "basketid", "orderid", "invoiceid", "invoiceno", "invoice"}]
        grouped_values = []
        for _, row in df[possible].iterrows():
            items = []
            for value in row.dropna().tolist():
                for product in re.split(r"[,;|]", str(value)):
                    product = product.strip()
                    if product and product.lower() not in {"nan", "none"}:
                        items.append(product)
            grouped_values.append(sorted(set(items)))
        grouped = pd.Series([x for x in grouped_values if x], name="items")
        source_format = "One transaction per row"
    grouped = grouped[grouped.map(len) > 0]
    if grouped.empty:
        raise ValueError("No usable transaction/product values were found. Check the column names and missing values.")
    grouped.index = range(1, len(grouped) + 1)
    transactions = pd.DataFrame({"Transaction": grouped.index, "items": grouped.values})
    quality = {"Detected format": source_format, "Source rows": len(df), "Transactions kept": len(transactions), "Rows with missing values": rows_missing, "Empty transactions removed": int(len(df) - len(transactions)) if source_format == "One transaction per row" else "Not applicable", "Duplicate product entries removed": "Repeated items within each basket are deduplicated."}
    return transactions, quality

def create_basket(transactions: pd.DataFrame) -> pd.DataFrame:
    all_items = sorted({item for items in transactions["items"] for item in items})
    records = [{item: item in items for item in all_items} for items in transactions["items"]]
    return pd.DataFrame(records, index=transactions["Transaction"]).astype(bool)

def format_itemset(itemset) -> str:
    return ", ".join(sorted(map(str, itemset)))
