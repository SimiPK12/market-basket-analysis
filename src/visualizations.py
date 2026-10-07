"""Plotly charts used in the dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .data_processing import format_itemset
COLORS = ["#176b69", "#2a8a7c", "#50a596", "#86bcae", "#b7d5ca"]

def product_chart(products, top_n=10):
    data = products.head(top_n).sort_values("Purchases")
    fig = px.bar(data, x="Purchases", y="Product", orientation="h", color="Purchases", color_continuous_scale=["#a8cec2", "#176b69"], title=f"Top {min(top_n,len(products))} products by basket count")
    fig.update_layout(template="plotly_white", coloraxis_showscale=False, xaxis_title="Transactions containing product", yaxis_title="", margin=dict(l=10,r=15,t=55,b=15), height=max(300, 34*len(data)))
    return fig

def itemset_chart(frequent):
    if frequent.empty:
        return go.Figure().update_layout(title="Frequent itemsets (none at current support)")
    data=frequent.assign(label=frequent.itemsets.map(format_itemset), size=frequent.itemsets.map(len)).sort_values("support",ascending=False).head(12).sort_values("support")
    fig=px.bar(data,x="support",y="label",orientation="h",color="size",color_continuous_scale="Teal",title="Top frequent itemsets by support",labels={"support":"Support","label":"Itemset","size":"Items"})
    fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=55,b=15),height=400)
    return fig

def rule_scatter(rules):
    fig=px.scatter(rules,x="confidence",y="lift",size="support",color="lift",hover_data=["Antecedents","Consequents","support"],color_continuous_scale="Teal",title="Rule confidence vs. lift",labels={"confidence":"Confidence","lift":"Lift","support":"Support"})
    fig.add_hline(y=1,line_dash="dash",line_color="#8797a3",annotation_text="Lift = 1 baseline")
    fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=55,b=15),height=400)
    return fig

def rule_chart(rules, top_n=10):
    data=rules.head(top_n).copy(); data["Rule"]=data["Antecedents"]+" → "+data["Consequents"]; data=data.sort_values("lift")
    fig=px.bar(data,x="lift",y="Rule",orientation="h",color="confidence",color_continuous_scale="Teal",hover_data=["support","confidence"],title="Highest-lift association rules",labels={"lift":"Lift","Rule":"Rule","confidence":"Confidence"})
    fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=55,b=15),height=max(330,35*len(data)))
    return fig

def cooccurrence_chart(basket, products, top_n=12):
    names=products.head(top_n)["Product"].tolist(); data=basket.reindex(columns=names,fill_value=False).astype(int)
    matrix=data.T.dot(data)
    fig=px.imshow(matrix,labels={"x":"Product","y":"Product","color":"Shared baskets"},color_continuous_scale="Teal",title=f"Co-occurrence among top {len(names)} products",aspect="auto")
    fig.update_layout(template="plotly_white",height=550,margin=dict(l=10,r=10,t=55,b=20))
    return fig
