import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set up page
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="centered"
)

# Tabs
tab1, tab2, tab3 = st.tabs(["📦 Product Recommendation", "📊 Customer Segmentation", "🎯 Predict Customer Segment"])

# ============================ TAB 1: Product Recommendation ============================
with tab1:
    st.title("📦 Product Recommendation Engine")
    st.markdown("Enter a product name and get similar product suggestions based on customer buying behavior.")

    # Load files
    try:
        with open("product_similarity.pkl", "rb") as f:
            product_similarity = pickle.load(f)
        with open("product_list.pkl", "rb") as f:
            product_list = pickle.load(f)
        with open("pivot_table.pkl", "rb") as f:
            pivot_table = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Required product files not found.")
        st.stop()

    product_to_index = {name: idx for idx, name in enumerate(product_list)}
    product_names = pivot_table.columns.tolist()

    # UI
    product_input = st.selectbox("Select Product Name", options=sorted(product_names))

    if st.button("Get Recommendations"):
        if product_input in product_to_index:
            index = product_to_index[product_input]
            similarities = product_similarity[index]
            recommended_indices = np.argsort(similarities)[-6:-1][::-1]

            st.write("### 🛒 Recommended Products:")
            for i in recommended_indices:
                st.write(product_list[i])
        else:
            st.error("❌ Product not found in dataset.")

# ============================ TAB 2: Customer Segmentation (View Clusters) ============================
with tab2:
    st.title("📊 Customer Segmentation using RFM Clustering")

    try:
        df = pd.read_csv("rfm_with_clusters.csv")
    except FileNotFoundError:
        st.error("❌ rfm_with_clusters.csv not found.")
        st.stop()

    st.subheader("📋 RFM Clustered Customer Data")
    st.dataframe(df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment', 'Cluster']].head(), use_container_width=True)

    st.subheader("📊 Customers per Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    cluster_counts.index = cluster_counts.index.astype(str)
    st.bar_chart(cluster_counts)

    st.subheader("📈 Cluster Profiles (Average RFM Scores)")
    cluster_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1).reset_index()
    st.table(cluster_summary)

# ============================ TAB 3: Predict Customer Segment ============================
with tab3:
    st.title("🎯 Predict Customer Segment")

    # Load model and scaler
    try:
        with open("rfm_cluster_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Required model or scaler file not found.")
        st.stop()

    # Input fields
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1, value=90)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, step=1, value=2)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0, value=500.0)

    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        scaled_input = scaler.transform(input_data)
        cluster = model.predict(scaled_input)[0]

        # Custom segment labels (you can edit as needed)
        segment_labels = {
            0: "Regular Shopper",
            1: "New/Low Value",
            2: "Occasional Shopper",
            3: "High-Value Loyal"
        }

        st.success(f"🧠 Predicted Cluster: {cluster}")
        st.info(f"This customer belongs to: **{segment_labels.get(cluster, 'Unknown')}**")
