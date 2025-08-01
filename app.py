import streamlit as st
import pandas as pd
import pickle

# Load models and data
try:
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    
    with open("product_name_mapping.pkl", "rb") as f:
        product_mapping = pickle.load(f)

    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)

    st.sidebar.success("✅ Models loaded successfully")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading models: {e}")

# App title and tabs
st.title("🛍️ Shopper Spectrum")
tabs = st.tabs(["Product Recommendation", "Customer Segmentation"])

# -----------------------------
# 🎯 1️⃣ Product Recommendation Module
# -----------------------------
with tabs[0]:
    st.header("🎯 Product Recommendation")
    product_input = st.text_input("Enter a Product Name")

    if st.button("Get Recommendations"):
        if product_input in product_mapping:
            product_index = product_mapping[product_input]
            similarity_scores = list(enumerate(kmeans_model[product_index]))
            sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
            recommended_products = [list(product_mapping.keys())[i] for i, _ in sorted_scores]

            st.subheader("🛒 Top 5 Recommended Products")
            for i, prod in enumerate(recommended_products, 1):
                st.markdown(f"**{i}.** {prod}")
        else:
            st.warning("Product not found. Please try a different name.")

# -----------------------------
# 🎯 2️⃣ Customer Segmentation Module
# -----------------------------
with tabs[1]:
    st.header("👥 Customer Segmentation")
    r = st.number_input("Recency (in days)", min_value=0)
    f = st.number_input("Frequency (number of purchases)", min_value=0)
    m = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame([[r, f, m]], columns=["Recency", "Frequency", "Monetary"])
        cluster = kmeans_model.predict(input_data)[0]

        label_map = {
            0: "At-Risk",
            1: "Occasional",
            2: "Regular",
            3: "High-Value"
        }
        label = label_map.get(cluster, f"Cluster {cluster}")
        st.success(f"🧠 Predicted Segment: **{label}**")
