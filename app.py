import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load Models ---
# Product recommendation model
with open("product_similarity.pkl", "rb") as f:
    product_similarity = pickle.load(f)

with open("product_list.pkl", "rb") as f:
    product_list = pickle.load(f)

# Customer segmentation model
with open("kmeans_model.pkl", "rb") as f:
    rfm_cluster_model = pickle.load(f)  # Keep the name same as original

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Streamlit App ---
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

st.title("🛍️ Shopper Spectrum")

tabs = st.tabs(["📦 Product Recommendation", "👥 Customer Segmentation", "🎯 Predict Customer Segment"])

# --- Tab 1: Product Recommendation ---
with tabs[0]:
    st.header("Product Recommendation")
    selected_product = st.selectbox("Select a product:", product_list)
    
    if st.button("Recommend"):
        index = product_list.index(selected_product)
        similarity_scores = list(enumerate(product_similarity[index]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_products = [product_list[i[0]] for i in sorted_scores[1:6]]
        
        st.subheader("Recommended Products:")
        for product in recommended_products:
            st.write(f"- {product}")

# --- Tab 2: Customer Segmentation Info ---
with tabs[1]:
    st.header("Customer Segmentation")
    st.write("This section provides information about how customers are segmented using RFM analysis and clustering.")

# --- Tab 3: Predict Customer Segment ---
with tabs[2]:
    st.header("Predict Customer Segment")
    
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spent)", min_value=0.0)
    
    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        scaled_data = scaler.transform(input_data)
        cluster_label = rfm_cluster_model.predict(scaled_data)[0]
        
        st.success(f"Predicted Customer Segment: {cluster_label}")
