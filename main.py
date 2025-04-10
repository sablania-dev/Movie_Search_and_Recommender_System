import streamlit as st
import pandas as pd
import os
from PIL import Image
from loading_and_preprocessing import (
    load_data,
    preprocess_data,
    load_collaborative_data,
    preprocess_collaborative_data,
    load_or_preprocess_data,
    load_or_preprocess_collaborative_data,
)
from functions1 import autocomplete, get_k_recommendations, get_collab_recommendation_score_for_all_movies, display_results_with_images

# Load and display the image with a custom size
st.image("images/Netflix.jpg", width=300)  # Replace with your image filename

# Login Section
st.subheader("Login")
user_id = st.text_input("User ID", placeholder="Enter your User ID")
password = st.text_input("Password", type="password", placeholder="Enter your Password")
login_button = st.button("Login")

if login_button:
    if password == "1234":  # Static password for all users
        st.success("Login successful!")
        
        # Load collaborative filtering data
        user_item_matrix = load_or_preprocess_collaborative_data()
        
        if int(user_id) in user_item_matrix.index:
            # Generate collaborative filtering recommendations
            svd_predictions = user_item_matrix.copy()  # Placeholder for SVD predictions
            recommendations = get_collab_recommendation_score_for_all_movies(user_item_matrix, int(user_id), svd_predictions)
            
            st.subheader("Recommended For You")
            # Display recommendations with images
            display_results_with_images(recommendations.head(10))
        else:
            st.error("User ID not found in the dataset.")
    else:
        st.error("Invalid User ID or Password.")

# Search bar with magnifying glass button
st.subheader("Search Movies")
user_input = st.text_input("Search", placeholder="Search for Movies...")
search_button = st.button("🔍 Search")

# Load Real DataFrame
data = load_or_preprocess_data()

# Display REAL search results when button is clicked
if search_button:
    if user_input:
        complete_input_string = autocomplete(data, user_input)
        filtered_df = get_k_recommendations(data, complete_input_string, 10)
        if not filtered_df.empty:
            st.subheader("Search Results")
            # Display search results with images
            display_results_with_images(filtered_df)
            # print(filtered_df.head())
        else:
            st.write("No results found.")
    else:
        st.write("Please enter a search term.")
