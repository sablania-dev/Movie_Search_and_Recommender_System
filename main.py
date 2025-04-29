import streamlit as st
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from loading_and_preprocessing import (
    load_or_preprocess_data,
    load_or_preprocess_collaborative_data,
)
from functions1 import autocomplete, get_k_recommendations, display_results_with_images, update_weights, get_content_recommendations, get_user_cf_recommendations, get_item_cf_recommendations

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Load data at the top so it is accessible throughout the script
data = load_or_preprocess_data()

# Load collaborative filtering data
user_item_matrix = load_or_preprocess_collaborative_data()

# Split the screen into three columns
col_left, col_middle, col_right = st.columns([1, 4, 1])


# Right Column: Configuration
with col_right:
    st.subheader("Configuration")
    st.write("Adjust the weights for scoring components:")
    
    # Initialize weights
    weights = {
        "Actor Score Weight": 0.25,
        "Genre Score Weight": 0.25,
        "Director Score Weight": 0.25,
        "Demographic Score Weight": 0.25,
    }

    # Create sliders dynamically
    for key in weights.keys():
        weights[key] = st.slider(
            key, 
            0.0, 
            1.0, 
            weights[key], 
            0.01, 
            on_change=lambda k=key: update_weights(weights, k, st.session_state[k]),
            key=key
        )

    # Normalize weights using softmax
    weights = update_weights(weights, None, None)

    # Extract normalized weights
    weight_actor = weights["Actor Score Weight"]
    weight_genre = weights["Genre Score Weight"]
    weight_director = weights["Director Score Weight"]
    weight_demographic = weights["Demographic Score Weight"]

    # Add a temperature slider
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

    # Display the current weights and temperature at the bottom
    st.write("### Current Weights")
    st.write(f"Actor Score Weight: {weight_actor:.2f}")
    st.write(f"Genre Score Weight: {weight_genre:.2f}")
    st.write(f"Director Score Weight: {weight_director:.2f}")
    st.write(f"Demographic Score Weight: {weight_demographic:.2f}")
    st.write(f"Temperature: {temperature:.1f}")



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

        if (int(user_id) in user_item_matrix.index) and password == "1234":  # Static password for all users
            st.success("Login successful!")
            st.write(f"Welcome, User {user_id}!")
            
            if int(user_id) in user_item_matrix.index:
                # Generate user-based collaborative filtering recommendations
                user_cf_recommendations = get_user_cf_recommendations(
                    user_item_matrix, 
                    int(user_id), 
                    temperature=temperature, 
                    n=10
                )
                
                st.subheader("Recommended For You: User-Based Collaborative Filtering")
                display_results_with_images(user_cf_recommendations)

                # Generate item-based collaborative filtering recommendations
                item_cf_recommendations = get_item_cf_recommendations(
                    user_item_matrix, 
                    int(user_id), 
                    temperature=temperature, 
                    n=10
                )
                
                st.subheader("Recommended For You: Item-Based Collaborative Filtering")
                display_results_with_images(item_cf_recommendations)

                # Generate content-based filtering recommendations
                content_recommendations = get_content_recommendations(
                    data, 
                    int(user_id), 
                    user_item_matrix, 
                    [weight_actor, weight_genre, weight_director, weight_demographic],
                    temperature=temperature
                )
                
                if not content_recommendations.empty:
                    st.subheader("Recommended For You: Content-Based Filtering")
                    display_results_with_images(content_recommendations)
                else:
                    st.write("No content-based recommendations available.")
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
                weights=[weight_actor, weight_genre, weight_director, weight_demographic]
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
    st.subheader("Debugging")
    st.write("Scoring Distribution")
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