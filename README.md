# Market Basket Analysis Using the Apriori Algorithm

A student-friendly Streamlit application for exploring grocery transactions, finding frequent itemsets, and evaluating association rules. The included sample contains 800 varied synthetic grocery baskets for an immediate offline demonstration after dependencies are installed. All metrics and results are computed from the selected CSV at runtime.

## Features

- Six sidebar sections: Dashboard, Data Exploration, Apriori Analysis, Association Rules, Visualizations, and About Project.
- Upload a CSV or use the included sample dataset.
- Accepts long transaction data (`TransactionID,Item`) and one-row-per-transaction data with comma-separated products (also supports multiple item columns).
- Cleans blanks and duplicate products per basket, then one-hot encodes transactions.
- Interactive minimum support, confidence, lift, and displayed-rule limits.
- Frequent itemsets, readable association rules, interpretations, CSV downloads, product frequencies, rule scatter plot, itemset chart, rule chart, and product co-occurrence heatmap.
- Local processing; no database or API key.

## Technologies

Python, Streamlit, Pandas, NumPy, MLxtend, Plotly.

## Folder structure

```text
Market_Basket_Analysis/
├── app.py                    # Streamlit UI, navigation, metrics and controls
├── requirements.txt          # Python dependencies
├── README.md                 # Setup, concepts, demo and viva notes
├── sample_data/
│   └── groceries.csv         # 800 generated demo transactions in long format
└── src/
    ├── __init__.py
    ├── data_processing.py    # CSV normalization, cleanup, basket encoding
    ├── apriori_analysis.py   # Frequent itemsets and association rules
    └── visualizations.py     # Reusable Plotly figures
```

## Dataset format

Preferred long format (one product per row):

```csv
TransactionID,Item
1,Milk
1,Bread
1,Butter
2,Bread
2,Jam
```

Transaction identifiers may also be named `OrderID`, `InvoiceID`, `Transaction`, or `BasketID`; product columns may be `Product`, `ProductName`, or `Description`. Alternatively, each row can be one basket, with products separated by commas in a cell, or spread across product columns. Empty values are dropped. Repeated products in the same basket are deduplicated.

## Installation and launch

Use Python 3.10 or later in a terminal opened at this project folder:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux, use `source .venv/bin/activate`. Install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Dependencies need internet access only for installation; the app itself does not call external services.

## How to use

1. Open the Dashboard; the sample is selected by default.
2. Review transaction and product counts and the top-products chart.
3. Use Data Exploration to inspect source rows, cleaned baskets, and quality notes.
4. Adjust support and confidence in the sidebar. Lift is an additional rule filter.
5. Review frequent itemsets and association rules; download either result as CSV.
6. Use Visualizations to compare product frequency, itemset support, and rule metrics.
7. Upload your own CSV from the sidebar to replace the sample.

## Apriori and rule metrics

Apriori finds itemsets that appear in at least the minimum fraction of transactions. Its downward-closure property says that every subset of a frequent itemset must also be frequent. This lets the algorithm prune candidate combinations early.

- **Support(A)** = transactions containing A / total transactions. It measures how common the itemset or combined rule is.
- **Confidence(A → B)** = Support(A ∪ B) / Support(A). It estimates the share of antecedent baskets that also contain the consequent.
- **Lift(A → B)** = Confidence(A → B) / Support(B). Lift above 1 means the consequent is more common with the antecedent than its overall baseline frequency; lift near 1 indicates little association.

Confidence is not a guarantee of future purchases. Association does not prove causation, and thresholds influence which patterns appear. Apriori can be expensive on very large, wide datasets; FP-Growth is a possible future comparison.

## Suggested faculty demonstration

1. Start on Dashboard and explain the goal: discover products that appear together in the actual transaction data.
2. Point to the dynamically calculated counts and top products.
3. Open Data Exploration; show the CSV rows and explain that repeated products inside one basket are collapsed.
4. Open Apriori Analysis; define support and show the current itemset table.
5. Lower or raise minimum support and show that the itemset count/table changes.
6. Open Association Rules; explain one generated rule using support, confidence, and lift. Lift above one indicates co-occurrence above the consequent's baseline.
7. Change the sort selector, then download association rules as CSV.
8. Open Visualizations and explain the confidence/lift scatter and co-occurrence heatmap.
9. Finish on About Project with applications, limitations, and future scope.

## Likely viva questions

**What is Market Basket Analysis?** A technique for discovering items that tend to occur together in transaction baskets.

**What is an itemset?** A set of one or more products evaluated for joint occurrence.

**What does Apriori do?** It finds itemsets meeting minimum support and uses frequent-subset logic to prune candidates.

**Why minimum support?** It avoids spending effort on combinations that are too rare to be useful.

**What is confidence?** The conditional frequency of the consequent among baskets containing the antecedent.

**What is lift?** Confidence divided by the consequent's baseline support; values over one indicate positive association.

**Does association imply causation?** No. It identifies co-occurrence, not why it happens.

**Why use Apriori?** It is a well-known, interpretable algorithm whose stages and measures are suitable for explaining in a student project.

**What are its limitations?** Candidate generation can be expensive on large datasets and results depend on thresholds and data quality.

**What can follow this project?** Compare FP-Growth, segment customers, or integrate recommendations into a retail system.

## Troubleshooting

- **CSV format error:** Use the column names and examples above. The app reports accepted layouts.
- **No frequent itemsets:** Lower minimum support.
- **No rules:** Lower minimum confidence/lift or support; rules require itemsets of size two or more.
- **Too many results:** Raise thresholds or reduce the display limit.
- **Import error:** Confirm the virtual environment is active and run `pip install -r requirements.txt` again.
- **Port in use:** Run `streamlit run app.py --server.port 8502`.
- **Empty input:** Select the sample toggle or upload a non-empty CSV.
