import streamlit as st
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from loading_and_preprocessing import (
    load_data,
    preprocess_data,
    load_collaborative_data,
    preprocess_collaborative_data,
    load_or_preprocess_data,
    load_or_preprocess_collaborative_data,
)
from functions1 import autocomplete, get_k_recommendations, get_collab_recommendation_score_for_all_movies, display_results_with_images

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Load data at the top so it is accessible throughout the script
data = load_or_preprocess_data()

# Split the screen into three columns
col_left, col_middle, col_right = st.columns([1, 4, 1])


# Right Column: Configuration
with col_right:
    st.subheader("Configuration")
    st.write("Adjust the weights for scoring components:")
    
    # Initialize weights
    weights = {
        "Actor Score Weight": 0.2,
        "Genre Score Weight": 0.2,
        "Keywords Score Weight": 0.2,
        "Director Score Weight": 0.2,
        "Demographic Score Weight": 0.2,
    }
    
    # Function to update weights dynamically
    def update_weights(changed_key, new_value):
        remaining_keys = [key for key in weights.keys() if key != changed_key]
        remaining_total = sum(weights[key] for key in remaining_keys)
        if remaining_total > 0:
            scale_factor = (1 - new_value) / remaining_total
            for key in remaining_keys:
                weights[key] *= scale_factor
        weights[changed_key] = new_value

    # Create sliders dynamically
    for key in weights.keys():
        weights[key] = st.slider(
            key, 
            0.0, 
            1.0, 
            weights[key], 
            0.01, 
            on_change=lambda k=key: update_weights(k, st.session_state[k]),
            key=key
        )
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        for key in weights.keys():
            weights[key] /= total_weight

    # Extract normalized weights
    weight_actor = weights["Actor Score Weight"]
    weight_genre = weights["Genre Score Weight"]
    weight_keywords = weights["Keywords Score Weight"]
    weight_director = weights["Director Score Weight"]
    weight_demographic = weights["Demographic Score Weight"]

    # Display the current weights at the bottom
    st.write("### Current Weights")
    st.write(f"Actor Score Weight: {weight_actor:.2f}")
    st.write(f"Genre Score Weight: {weight_genre:.2f}")
    st.write(f"Keywords Score Weight: {weight_keywords:.2f}")
    st.write(f"Director Score Weight: {weight_director:.2f}")
    st.write(f"Demographic Score Weight: {weight_demographic:.2f}")



# Middle Column: Search and Recommendations (already implemented)
with col_middle:
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

    st.subheader("Search Movies")
    user_input = st.text_input("Search", placeholder="Search for Movies...")
    search_button = st.button("🔍 Search")

    # Display REAL search results when button is clicked
    if search_button:
        if user_input:
            complete_input_string = autocomplete(data, user_input)
            filtered_df = get_k_recommendations(
                data, 
                complete_input_string, 
                10, 
                weights=[weight_actor, weight_genre, weight_keywords, weight_director, weight_demographic]
            )
            if not filtered_df.empty:
                st.subheader("Search Results")
                # Display search results with images
                display_results_with_images(filtered_df)
            else:
                st.write("No results found.")
        else:
            st.write("Please enter a search term.")


# Left Column: Scoring Distribution
with col_left:
    st.subheader("Scoring Distribution")
    images_folder = os.path.join(os.getcwd(), "images")
    os.makedirs(images_folder, exist_ok=True)
    
    # Plot histogram for weighted_score
    if 'weighted_score' in data.columns:
        fig, ax = plt.subplots()
        ax.hist(data['weighted_score'], bins=20, color='blue', alpha=0.7)
        ax.set_title("Weighted Score")
        ax.set_xlabel("Score")
        ax.set_ylabel("Number of Movies")
        histogram_path = os.path.join(images_folder, "weighted_score_histogram.png")
        plt.savefig(histogram_path)
        plt.close(fig)
        st.image(histogram_path, caption="Weighted Score Histogram")
    
    # Plot histograms for normalized scores
    for col in ['norm_actor_score', 'norm_genre_score', 'norm_kwd_score', 'norm_diro_score', 'norm_dmg_score']:
        if col in data.columns:
            fig, ax = plt.subplots()
            ax.hist(data[col], bins=20, color='green', alpha=0.7)
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlabel("Score")
            ax.set_ylabel("Number of Movies")
            histogram_path = os.path.join(images_folder, f"{col}_histogram.png")
            plt.savefig(histogram_path)
            plt.close(fig)
            st.image(histogram_path, caption=f"{col.replace('_', ' ').title()} Histogram")