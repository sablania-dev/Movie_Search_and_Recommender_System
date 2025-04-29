import streamlit as st
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from loading_and_preprocessing import (
    load_or_preprocess_data,
    load_or_preprocess_collaborative_data,
)
from functions1 import display_results_with_images, update_weights, get_content_recommendations, get_user_cf_recommendations, get_item_cf_recommendations
from individual_recommenders import (
    popular_right_now,
    top_grossing,
    critically_acclaimed,
    hidden_gems,
)

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Load data at the top so it is accessible throughout the script
data = load_or_preprocess_data()

# Load collaborative filtering data
user_item_matrix = load_or_preprocess_collaborative_data()

# Split the screen into two columns (removing the left column)
col_middle, col_right = st.columns([4, 1])


# Right Column: Configuration
with col_right:
    st.subheader("Configuration")
    st.write("Adjust the weights for scoring components:")
    
    # Add a temperature slider (display 0 to 1, map to 0.1 to 0.5 internally)
    display_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.01)
    temperature = 0.1 + (display_temperature * 0.4)  # Map 0-1 to 0.1-0.5

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


    # Display the current weights and temperature at the bottom
    st.write("### Current Weights")
    st.write(f"Actor Score Weight: {weight_actor:.2f}")
    st.write(f"Genre Score Weight: {weight_genre:.2f}")
    st.write(f"Director Score Weight: {weight_director:.2f}")
    st.write(f"Demographic Score Weight: {weight_demographic:.2f}")
    st.write(f"Temperature (Real Value): {temperature:.2f}")



# Middle Column: Recommendations
with col_middle:
    # Load and display the image with a custom size
    st.image("images/Netflix.jpg", width=300)  # Replace with your image filename

    # Login Section
    st.subheader("Login")
    user_id = st.text_input("User ID", placeholder="Enter your User ID")
    password = st.text_input("Password", type="password", placeholder="Enter your Password")

    # Dropdown for selecting recommender type
    recommender_type = st.selectbox(
        "Select Recommender Type",
        [
            "User-Based Collaborative Filtering",
            "Item-Based Collaborative Filtering",
            "Content-Based Filtering",
            "Popular Right Now",
            "Top Grossing",
            "Critically Acclaimed",
            "Hidden Gems",
        ]
    )

    # Button to show recommendations
    show_recommendations_button = st.button("Show Recommendations")

    if show_recommendations_button:
        if (int(user_id) in user_item_matrix.index) and password == "1234":  # Static password for all users
            st.success("Login successful!")
            st.write(f"Welcome, User {user_id}!")

            if int(user_id) in user_item_matrix.index:
                # Generate recommendations based on the selected type
                if recommender_type == "User-Based Collaborative Filtering":
                    user_cf_recommendations = get_user_cf_recommendations(
                        user_item_matrix, 
                        int(user_id), 
                        temperature=temperature, 
                        n=10
                    )
                    st.subheader("Recommended For You: User-Based Collaborative Filtering")
                    display_results_with_images(user_cf_recommendations)

                elif recommender_type == "Item-Based Collaborative Filtering":
                    item_cf_recommendations = get_item_cf_recommendations(
                        user_item_matrix, 
                        int(user_id), 
                        temperature=temperature, 
                        n=10
                    )
                    st.subheader("Recommended For You: Item-Based Collaborative Filtering")
                    display_results_with_images(item_cf_recommendations)

                elif recommender_type == "Content-Based Filtering":
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

                elif recommender_type == "Popular Right Now":
                    popular_recommendations = popular_right_now(data, temperature=temperature, n=10)
                    st.subheader("Recommended For You: Popular Right Now")
                    display_results_with_images(popular_recommendations)

                elif recommender_type == "Top Grossing":
                    top_grossing_recommendations = top_grossing(data, temperature=temperature, n=10)
                    st.subheader("Recommended For You: Top Grossing")
                    display_results_with_images(top_grossing_recommendations)

                elif recommender_type == "Critically Acclaimed":
                    critically_acclaimed_recommendations = critically_acclaimed(data, temperature=temperature, n=10)
                    st.subheader("Recommended For You: Critically Acclaimed")
                    display_results_with_images(critically_acclaimed_recommendations)

                elif recommender_type == "Hidden Gems":
                    hidden_gems_recommendations = hidden_gems(data, temperature=temperature, n=10)
                    st.subheader("Recommended For You: Hidden Gems")
                    display_results_with_images(hidden_gems_recommendations)

            else:
                st.error("User ID not found in the dataset.")
        else:
            st.error("Invalid User ID or Password.")