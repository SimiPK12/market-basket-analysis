"""Streamlit interface for an explainable market basket analysis project."""
from pathlib import Path
import streamlit as st
import pandas as pd
from src.data_processing import load_data, preprocess_transactions, create_basket, format_itemset
from src.apriori_analysis import run_apriori, generate_rules
from src.visualizations import product_chart, itemset_chart, rule_scatter, rule_chart, cooccurrence_chart

st.set_page_config(page_title="Market Basket Analysis", page_icon="▦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
html,body,[class*="css"]{font-family:Inter,'Segoe UI',sans-serif} h1,h2,h3{font-family:Inter,'Segoe UI',sans-serif;letter-spacing:-.035em}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1440px}
[data-testid="stMetric"]{background:linear-gradient(135deg,#fff,#f5f8fc);border:1px solid #e7edf5;padding:18px 20px;border-radius:14px;box-shadow:0 5px 18px #1e3a5f0b}
[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricLabel"],[data-testid="stMetric"] [data-testid="stMetricValue"],[data-testid="stMetric"] [data-testid="stMetricDelta"]{color:#183246!important}
[data-testid="stMetric"] *{color:#183246!important}
.hero{background:linear-gradient(115deg,#102c46,#1d5261 64%,#287b72);color:white;padding:34px 38px;border-radius:18px;margin-bottom:20px}
.hero h1{color:white;margin:0 0 7px}.hero p{color:#d5e8e9;font-size:1.05rem;margin:0}
.note{padding:14px 17px;background:#f1f6fa;border-left:3px solid #238477;border-radius:7px;color:#355064;margin:14px 0}
.step{background:#f5f8fb;border:1px solid #e5edf3;border-radius:12px;padding:15px;text-align:center;font-weight:600;color:#244254}
</style>""", unsafe_allow_html=True)

SAMPLE = Path(__file__).parent / "sample_data" / "groceries.csv"
st.sidebar.markdown("## MARKET BASKET")
page = st.sidebar.radio("Navigate", ["Dashboard", "Data Exploration", "Apriori Analysis", "Association Rules", "Visualizations", "About Project"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("### Transaction data")
uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], help="Supports TransactionID, Item rows or one transaction per row with comma-separated products.")
use_sample = st.sidebar.toggle("Use included grocery sample", value=uploaded is None)
st.sidebar.caption("Your data is processed locally in this session.")
st.sidebar.divider()
st.sidebar.markdown("### Apriori settings")
if st.sidebar.button("Reset demo thresholds", use_container_width=True):
    st.session_state["min_support"] = 0.03
    st.session_state["min_confidence"] = 0.30
    st.session_state["min_lift"] = 1.0
    st.session_state["max_rules"] = 25
support = st.sidebar.slider("Minimum support", 0.001, 0.20, 0.03, 0.001, format="%.3f", key="min_support")
confidence = st.sidebar.slider("Minimum confidence", 0.10, 1.00, 0.30, 0.05, format="%.2f", key="min_confidence")
min_lift = st.sidebar.slider("Minimum lift", 0.0, 5.0, 1.0, 0.1, format="%.1f", key="min_lift")
max_rules = st.sidebar.slider("Maximum rules to display", 5, 100, 25, 5, key="max_rules")

def empty_state(message):
    st.info(message)

@st.cache_data(show_spinner=False)
def read_source(file_bytes, source_name):
    return load_data(file_bytes, source_name)

try:
    if uploaded is not None:
        raw = read_source(uploaded.getvalue(), uploaded.name)
        source_label = uploaded.name
    elif use_sample:
        raw = read_source(SAMPLE.read_bytes(), "groceries.csv")
        source_label = "Included grocery sample"
    else:
        raw = None
        source_label = "No dataset selected"
    if raw is not None:
        transactions, quality = preprocess_transactions(raw)
        basket = create_basket(transactions)
        products = transactions.explode("items")["items"].value_counts().rename_axis("Product").reset_index(name="Purchases")
        frequent = run_apriori(basket, support)
        rules_all = generate_rules(frequent, confidence) if not frequent.empty else pd.DataFrame()
        if not rules_all.empty:
            rules_all["Antecedents"] = rules_all["antecedents"].map(format_itemset)
            rules_all["Consequents"] = rules_all["consequents"].map(format_itemset)
            rules_all = rules_all[rules_all["lift"] >= min_lift].copy()
            rules_all["Interpretation"] = rules_all.apply(lambda r: f"When a customer buys {r['Antecedents']}, they are likely to also buy {r['Consequents']}.", axis=1)
            rules_all = rules_all.sort_values("lift", ascending=False).reset_index(drop=True)
        frequent_view = frequent.copy()
        if not frequent_view.empty:
            frequent_view["Itemsets"] = frequent_view["itemsets"].map(format_itemset)
            frequent_view["Itemset Size"] = frequent_view["itemsets"].map(len)
        st.sidebar.caption(f"Source: {source_label} · {len(transactions):,} transactions")
    else:
        raw = None
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"The CSV could not be understood: {exc}")
    st.info("Expected formats: columns named TransactionID and Item (one product per row), or one transaction per row with comma-separated product names.")
    raw = None
except Exception as exc:
    st.error(f"Analysis could not be prepared: {exc}")
    raw = None

def require_data():
    if raw is None:
        empty_state("Choose the included sample or upload a valid CSV from the sidebar to begin.")
        return False
    return True

def show_rules_table(frame, limit=None):
    if frame.empty:
        empty_state("No association rules meet these thresholds. Reduce minimum support, confidence, or lift.")
        return
    shown = frame.head(limit) if limit else frame
    display = shown[["Antecedents", "Consequents", "support", "confidence", "lift"]].rename(columns={"support":"Support", "confidence":"Confidence", "lift":"Lift"})
    st.dataframe(display.style.format({"Support":"{:.3f}","Confidence":"{:.3f}","Lift":"{:.2f}"}).background_gradient(subset=["Lift"], cmap="YlGn"), use_container_width=True, hide_index=True)

if page == "Dashboard":
    st.markdown('<div class="hero"><h1>Market Basket Analysis</h1><p>Discovering customer purchasing patterns with the Apriori algorithm</p></div>', unsafe_allow_html=True)
    st.write("Market Basket Analysis is a data mining technique used to discover relationships between products frequently purchased together. Apriori identifies frequent itemsets and derives interpretable association rules.")
    if require_data():
        cols = st.columns(4)
        for col, label, value, help_text in zip(cols, ["Transactions","Unique products","Frequent itemsets","Association rules"], [len(transactions), len(products), len(frequent), len(rules_all)], ["Distinct baskets after cleaning","Distinct product names","At the selected minimum support","After selected confidence and lift filters"]):
            col.metric(label, f"{value:,}", help=help_text)
        left, right = st.columns([1.1, .9])
        with left:
            st.subheader("Most purchased products")
            st.plotly_chart(product_chart(products, 10), use_container_width=True)
        with right:
            st.subheader("Analysis pipeline")
            st.markdown("<div class='step'>Dataset → Cleaning → Basket encoding</div><br><div class='step'>Apriori → Frequent itemsets</div><br><div class='step'>Association rules → Insights</div>", unsafe_allow_html=True)
            st.markdown("<div class='note'>Start with Data Exploration, then adjust the thresholds under Apriori settings to see how the results change.</div>", unsafe_allow_html=True)
    else:
        require_data()
elif page == "Data Exploration":
    st.title("Data Exploration")
    st.caption("Understand the transaction structure and data quality before mining patterns.")
    if require_data():
        a,b,c = st.columns(3); a.metric("Rows in source", f"{raw.shape[0]:,}"); b.metric("Transactions", f"{len(transactions):,}"); c.metric("Unique products", f"{len(products):,}")
        st.subheader("Source preview"); st.dataframe(raw.head(15), use_container_width=True, hide_index=True)
        st.subheader("Most frequently purchased products"); st.plotly_chart(product_chart(products, 15), use_container_width=True)
        st.caption("Counts measure the number of distinct baskets containing each product; repeated rows within the same basket are removed.")
        with st.expander("Data quality and processed baskets"):
            st.write(quality)
            st.dataframe(transactions.head(20).assign(items=transactions.head(20)["items"].map(lambda xs: ", ".join(xs))), use_container_width=True, hide_index=True)
            st.download_button("Download processed transactions", transactions.assign(items=transactions["items"].map(lambda xs: ", ".join(xs))).to_csv(index=False).encode(), "processed_transactions.csv", "text/csv")
elif page == "Apriori Analysis":
    st.title("Apriori Analysis")
    st.caption("Tune minimum support in the sidebar. The result table updates from the currently selected dataset and threshold.")
    st.markdown("<div class='note'><b>Support</b> = transactions containing an itemset ÷ total transactions. Apriori prunes candidate itemsets whose subsets are not frequent.</div>", unsafe_allow_html=True)
    if require_data():
        st.markdown(f"**Minimum support:** {support:.3f} &nbsp; · &nbsp; **Frequent itemsets found:** {len(frequent):,}")
        if frequent.empty: empty_state("No frequent itemsets were found. Try reducing the minimum support.")
        else:
            view = frequent_view[["Itemsets","support","Itemset Size"]].rename(columns={"support":"Support"}).sort_values("Support",ascending=False)
            st.dataframe(view.style.format({"Support":"{:.3f}"}), use_container_width=True, hide_index=True)
            st.download_button("Download frequent itemsets", view.to_csv(index=False).encode(), "frequent_itemsets.csv", "text/csv")
        with st.expander("Measures and formulas"):
            st.markdown("**Support(A)** = transactions containing A / total transactions  \n**Confidence(A → B)** = Support(A ∪ B) / Support(A)  \n**Lift(A → B)** = Confidence(A → B) / Support(B)")
            st.write("Confidence is the conditional purchase rate. Lift compares that rate with the consequent's baseline frequency: lift above 1 indicates a positive association; near 1 indicates little association.")
elif page == "Association Rules":
    st.title("Association Rules")
    st.caption("Rules are calculated from the current frequent itemsets and confidence threshold.")
    if require_data():
        if not rules_all.empty:
            sort_by = st.selectbox("Sort rules by", ["Lift", "Confidence", "Support"])
            sort_col = {"Lift":"lift", "Confidence":"confidence", "Support":"support"}[sort_by]
            display_rules = rules_all.sort_values(sort_col, ascending=False)
            st.subheader("Top strongest rules"); show_rules_table(display_rules, 10)
            st.subheader("All matching rules"); show_rules_table(display_rules, max_rules)
            st.download_button("Download association rules as CSV", display_rules[["Antecedents","Consequents","support","confidence","lift","Interpretation"]].to_csv(index=False).encode(), "association_rules.csv", "text/csv")
            with st.expander("How to interpret a rule"):
                r = display_rules.iloc[0]
                st.write(r["Interpretation"])
                st.write(f"Support {r['support']:.3f}: both sides occur in this fraction of all baskets. Confidence {r['confidence']:.3f}: this fraction of baskets containing the antecedent also contain the consequent. Lift {r['lift']:.2f}: co-purchase is {r['lift']:.2f}× the consequent's baseline rate.")
        else:
            st.info(f"No association rules match the current settings (support {support:.3f}, confidence {confidence:.2f}, lift {min_lift:.1f}). Lower one or more thresholds, or use **Reset demo thresholds** in the sidebar to restore settings that produce rules for the included sample.")
elif page == "Visualizations":
    st.title("Visualizations")
    st.caption(f"Charts use the current dataset and settings: support ≥ {support:.3f}, confidence ≥ {confidence:.2f}, lift ≥ {min_lift:.1f}. Each chart answers a different analytical question about the observed baskets and mined rules.")
    if require_data():
        left,right=st.columns(2)
        with left:
            st.plotly_chart(product_chart(products,10),use_container_width=True)
            st.plotly_chart(itemset_chart(frequent),use_container_width=True)
        with right:
            if not rules_all.empty:
                st.plotly_chart(rule_scatter(rules_all),use_container_width=True)
                st.plotly_chart(rule_chart(rules_all,10),use_container_width=True)
            else: empty_state("No rules available to visualize; adjust the thresholds.")
        with st.expander("Product co-occurrence heatmap"):
            st.plotly_chart(cooccurrence_chart(basket, products, 12),use_container_width=True)
elif page == "About Project":
    st.title("About the Project")
    st.markdown("### Problem statement\nRetailers collect transaction records but need a systematic way to discover which products tend to appear together. This project applies association-rule mining to transaction data and presents measurable, explainable patterns.")
    st.markdown("### Objectives\n- Identify frequently purchased products and item combinations.\n- Generate and evaluate association rules using support, confidence and lift.\n- Explore how thresholds change the number and strength of discovered patterns.")
    st.markdown("### Algorithm and technologies\n**Apriori** uses the downward-closure property: if an itemset is frequent, all its subsets must also be frequent. Implemented in Python with Pandas, NumPy, MLxtend, Streamlit and Plotly.")
    st.markdown("### Applications\nRetail analytics · Cross-selling · Product placement · E-commerce recommendations · Promotional planning")
    st.markdown("### Limitations\nApriori can be computationally expensive on very large datasets. Results depend on support and confidence thresholds. Association indicates co-occurrence, not causation.")
    st.markdown("### Future scope\nCompare with FP-Growth, add customer-segment analysis, explore personalized and real-time recommendations, and connect to larger retail data sources.")
