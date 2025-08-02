import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
@st.cache_data
def load_similarity():
    return pickle.load(open("product_similarity.pkl", "rb"))

@st.cache_data
def load_kmeans_model():
    return pickle.load(open("kmeans_model.pkl", "rb"))

@st.cache_data
def load_rfm_segmented():
    return pd.read_csv("rfm_segmented.csv")

@st.cache_data
def load_data():
    return pd.read_csv("rfm_with_clusters.csv")

# Load assets
similarity = load_similarity()
rfm_segmented = load_rfm_segmented()
rfm_data = load_data()
kmeans_model = load_kmeans_model()
products = rfm_data['Description'].dropna().unique().tolist()

# App title
st.set_page_config(page_title="🛍 Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum")
st.markdown("**An E-Commerce Customer Segmentation and Recommendation App**")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Product Recommendation", "Customer Segmentation"])

# -----------------------------------
# 1️⃣ PRODUCT RECOMMENDATION MODULE
# -----------------------------------
if option == "Product Recommendation":
    st.header("🔍 Product Recommendation Engine")
    product_name = st.text_input("Enter a product name:", "")

    if product_name:
        if product_name not in products:
            st.error("❌ Product not found in database. Please try another.")
        else:
            index = list(products).index(product_name)
            distances = similarity[index]
            recommended_products = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

            st.success(f"✅ Top 5 recommendations for **{product_name}**:")
            for i, (idx, score) in enumerate(recommended_products, start=1):
                st.markdown(f"{i}. {products[idx]}  — Similarity Score: `{score:.3f}`")

# -----------------------------------
# 2️⃣ CUSTOMER SEGMENTATION MODULE
# -----------------------------------
elif option == "Customer Segmentation":
    st.header("👥 Customer Segmentation Engine")

    st.markdown("Upload customer Recency, Frequency, and Monetary (RFM) data to predict customer segment.")
    uploaded_file = st.file_uploader("📂 Upload a CSV file with RFM columns", type=["csv"])

    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)

        required_cols = {'Recency', 'Frequency', 'Monetary'}
        if not required_cols.issubset(user_data.columns):
            st.error("❌ The file must contain Recency, Frequency, and Monetary columns.")
        else:
            # Predict clusters
            user_data_cleaned = user_data[['Recency', 'Frequency', 'Monetary']]
            predicted_clusters = kmeans_model.predict(user_data_cleaned)
            user_data['Cluster'] = predicted_clusters

            # Merge with labels
            labels_map = rfm_segmented[['Cluster', 'Segment']].drop_duplicates()
            user_data = user_data.merge(labels_map, on='Cluster', how='left')

            st.success("✅ Segmentation complete!")
            st.dataframe(user_data)

            csv = user_data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Segmentation Results", csv, "segmented_customers.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("🧠 Built with Streamlit | 💼 Developed by Gayatri Khairnar")
