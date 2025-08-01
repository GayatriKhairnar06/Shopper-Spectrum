import streamlit as st
import pandas as pd
import pickle
import numpy as np
# Load similarity matrix
with open("product_similarity.pkl", "rb") as f:
    product_similarity = pickle.load(f)
product_to_index = {name: idx for idx, name in enumerate(product_list)}
# Load pivot table (to validate product names)
with open("pivot_table.pkl", "rb") as f:
    pivot_table = pickle.load(f)
with open("product_list.pkl", "rb") as f:
    product_list = pickle.load(f)

# List of all product names
product_names = pivot_table.columns.tolist()

# Streamlit UI
st.title("🛍️ Product Recommendation Engine")
st.markdown("Enter a product name and get similar product suggestions based on customer buying behavior.")

# Input box
product_input = st.text_input("Enter Product Name")

# Recommend button
if st.button("Get Recommendations"):
    if product_input in product_names:
        # Get similarity scores for the input product
        similarities = product_similarity[product_input]
        similar_products = similarities.sort_values(ascending=False)[1:6]  # Exclude the product itself

        st.subheader("🔎 Top 5 Similar Products:")
        for i, (prod, score) in enumerate(similar_products.items(), start=1):
            st.write(f"{i}. **{prod}** (Similarity: {score:.2f})")
    else:
        st.error("❌ Product not found. Please check the spelling or try another product.")
