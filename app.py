import streamlit as st
import pandas as pd
import pickle
import numpy as np
st.set_page_config(
    page_title="Shopper Spectrum: Product Recommendation",
    page_icon="🛍️",  # Optional: Emoji for tab icon
    layout="centered"
)


# ---- Load files ----
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

# ---- Mapping ----
product_to_index = {name: idx for idx, name in enumerate(product_list)}
product_names = pivot_table.columns.tolist()

# ---- Streamlit UI ----
st.title("🛍️ Product Recommendation Engine")
st.markdown("Enter a product name and get similar product suggestions based on customer buying behavior.")

# Dropdown to avoid spelling errors
product_input = st.selectbox("Select Product Name", options=sorted(product_names))

# ---- Recommend button ----
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

