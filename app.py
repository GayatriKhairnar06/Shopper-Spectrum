import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---- Page Config ----
st.set_page_config(
    page_title="Shopper Spectrum: Product Recommendation",
    page_icon="🛍️", 
    layout="wide"
)

# ---- Load Files ----
try:
    with open("product_similarity.pkl", "rb") as f:
        product_similarity = pickle.load(f)
except FileNotFoundError:
    st.error("❌ product_similarity.pkl not found.")
    st.stop()

try:
    with open("product_list.pkl", "rb") as f:
        product_list = pickle.load(f)
except FileNotFoundError:
    st.error("❌ product_list.pkl not found.")
    st.stop()

try:
    with open("pivot_table.pkl", "rb") as f:
        pivot_table = pickle.load(f)
except FileNotFoundError:
    st.error("❌ pivot_table.pkl not found.")
    st.stop()

# ---- Create Mapping ----
product_to_index = {name: idx for idx, name in enumerate(product_list)}
product_names = pivot_table.columns.tolist()

# ---- Tabs UI ----
tab1, tab2 = st.tabs(["🛍️ Product Recommendation", "👥 Customer Segmentation"])

# =========================
#     TAB 1: RECOMMENDER
# =========================
with tab1:
    st.title("🛍️ Product Recommendation Engine")
    st.markdown("Enter a product name and get similar product suggestions based on customer buying behavior.")

    # Dropdown to select product
    product_input = st.selectbox("Select Product Name", options=sorted(product_names))

    if st.button("Get Recommendations"):
        if product_input in product_to_index:
            index = product_to_index[product_input]
            similarities = product_similarity[index]
            recommended_indices = np.argsort(similarities)[-6:-1][::-1]

            st.write("### Recommended Products:")
            for i in recommended_indices:
                st.write(product_list[i])
        else:
            st.error("❌ Product not found in the dataset.")

# =========================
#     TAB 2: CLUSTERING
# =========================
with tab2:
    st.title("👥 Customer Segmentation using RFM Clustering")

    try:
        df = pd.read_csv("rfm_with_cluster.csv")
    except FileNotFoundError:
        st.error("❌ rfm_with_cluster.csv not found.")
        st.stop()

    st.subheader("📋 RFM Clustered Customer Data")
    st.dataframe(df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'Cluster']].head(), use_container_width=True)

    st.subheader("📊 Customers per Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.astype(str)  # Ensure X-axis labels are visible
    st.bar_chart(cluster_counts)

    st.subheader("📈 Cluster Profiles (Average RFM Scores)")
    cluster_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1).reset_index()
    st.table(cluster_summary)



