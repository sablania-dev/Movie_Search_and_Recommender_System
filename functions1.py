import difflib
import pandas as pd
from individual_recommenders import svd_recommender, cosine_recommender, demographic_filtering
import os
from PIL import Image
import streamlit as st
import numpy as np
import time  # Import time module for debugging

def weighted_score(scores, weights):
    """
    Calculates the weighted score given a list of scores and corresponding weights.
    
    :param scores: List of scores (each between 0 and 1).
    :param weights: List of weights (each between 0 and 1).
    :return: Weighted score (between 0 and 1).
    """
    if len(scores) != len(weights):
        raise ValueError("Scores and weights must have the same number of elements.")

    # Ensure inputs are scalar values
    scores = [float(s) if isinstance(s, (int, float)) else 0 for s in scores]
    weights = [float(w) if isinstance(w, (int, float)) else 0 for w in weights]

    if not (all(0 <= s <= 1 for s in scores) and all(0 <= w <= 1 for w in weights)):
        raise ValueError("All scores and weights must be between 0 and 1.")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0  # Avoid division by zero, return a neutral score

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


def normalize_scores(series, mean=0.5, std=0.4):
    """
    Normalizes a pandas Series using standardization with a default mean of 0.5 and std of 0.4.
    Values outside the range [0, 1] are clipped to the limits.
    
    Parameters:
    series (pd.Series): The series to normalize.
    mean (float): The target mean for normalization.
    std (float): The target standard deviation for normalization.
    
    Returns:
    pd.Series: The normalized series.
    """
    normalized = (series - series.mean()) / series.std() * std + mean
    return normalized.clip(0, 1)  # Clip values to the range [0, 1]

def display_results_with_images(results_df):
    """
    Displays results horizontally with images and titles.
    Shows movie metadata on the right side of each image, including weighted score, year of release, and other details.
    If an image does not exist, a blank template is shown.
    If required columns are missing, fetches data from preprocessed_movies.csv.
    """
    current_directory = os.getcwd()
    images_folder = os.path.join(current_directory, "images")
    preprocessed_file = os.path.join(current_directory, "data", "preprocessed_movies.csv")

    # Load preprocessed data if required columns are missing
    if not {'release_date', 'id'}.issubset(results_df.columns):
        if os.path.exists(preprocessed_file):
            preprocessed_data = pd.read_csv(preprocessed_file)
            results_df = results_df.merge(preprocessed_data[['id', 'release_date']], on='id', how='left')

    for index, row in results_df.iterrows():
        # Use the index as the movie ID
        movie_id = index
        title = row['title']
        image_path = os.path.join(images_folder, f"{movie_id}.jpg")

        # Extract year of release from release_date
        release_year = None
        if 'release_date' in row and not pd.isna(row['release_date']):
            release_year = pd.to_datetime(row['release_date'], errors='coerce').year

        # Check if the image exists
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            # Create a blank template if the image does not exist
            image = Image.new('RGB', (200, 300), color='gray')

        # Display the image and metadata side by side
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=f"{movie_id}: {title}", width=150)
        with col2:
            st.write(f"**Title:** {title}")
            if release_year:
                st.write(f"**Year of Release:** {release_year}")
            if 'weighted_score' in row and not pd.isna(row['weighted_score']):
                st.write(f"**Weighted Score:** {row['weighted_score']:.2f}")
            if 'vote_average' in row and not pd.isna(row['vote_average']):
                st.write(f"**Rating:** {row['vote_average']}")
            if 'vote_count' in row and not pd.isna(row['vote_count']):
                st.write(f"**Votes:** {row['vote_count']}")
            if 'genres' in row and isinstance(row['genres'], list):
                st.write(f"**Genres:** {', '.join(row['genres'])}")
            if 'director' in row and not pd.isna(row['director']):
                st.write(f"**Director:** {row['director']}")

def update_weights(weights, changed_key, new_value):
    """
    Updates weights dynamically using the softmax function to ensure they sum to 1.
    
    Parameters:
    weights (dict): Dictionary of weights with keys as component names and values as current weights.
    changed_key (str): The key of the weight that was changed.
    new_value (float): The new value for the changed weight.
    
    Returns:
    dict: Updated weights normalized using the softmax function.
    """
    # Update the changed weight
    if changed_key is not None:
        weights[changed_key] = new_value

    # Ensure all values are numeric
    numeric_values = np.array([float(value) for value in weights.values()])
    
    # Apply softmax normalization
    softmax_values = np.exp(numeric_values) / np.sum(np.exp(numeric_values))
    
    # Return updated weights as a dictionary
    return dict(zip(weights.keys(), softmax_values))

def get_user_cf_recommendations(user_item_matrix, user_id, temperature=1.0, n=10):
    """
    Generates user-based collaborative filtering recommendations for a user.
    
    Parameters:
    user_item_matrix (pd.DataFrame): User-item matrix with user ratings.
    user_id (int): User ID for whom recommendations are to be generated.
    temperature (float): Temperature parameter for sampling.
    n (int): Number of recommendations to return.
    
    Returns:
    pd.DataFrame: DataFrame containing sampled n recommended movies with normalized scores.
    """
    user_similarity = cosine_recommender(user_item_matrix)
    similar_users = user_similarity[user_id].drop(user_id)
    
    # Align indices of similar_users with user_item_matrix columns
    similar_users = similar_users.reindex(user_item_matrix.index, fill_value=0)
    
    user_cf_scores = user_item_matrix.T.dot(similar_users).sort_values(ascending=False)
    user_cf_recommendations = pd.DataFrame({'title': user_cf_scores.index, 'score': user_cf_scores.values})
    
    # Normalize scores to range [0, 1]
    user_cf_recommendations['score'] = (user_cf_recommendations['score'] - user_cf_recommendations['score'].min()) / \
                                       (user_cf_recommendations['score'].max() - user_cf_recommendations['score'].min())
    
    # Sample recommendations based on temperature
    user_cf_recommendations = user_cf_recommendations.sample(
        n=n, 
        weights=(user_cf_recommendations['score'] ** (1 / temperature))
    )
    
    return user_cf_recommendations

def get_item_cf_recommendations(user_item_matrix, user_id, temperature=1.0, n=10):
    """
    Generates item-based collaborative filtering recommendations for a user.
    
    Parameters:
    user_item_matrix (pd.DataFrame): User-item matrix with user ratings.
    user_id (int): User ID for whom recommendations are to be generated.
    temperature (float): Temperature parameter for sampling.
    n (int): Number of recommendations to return.
    
    Returns:
    pd.DataFrame: DataFrame containing sampled n recommended movies with normalized scores.
    """
    # Apply SVD to generate predictions
    svd_predictions = svd_recommender(user_item_matrix)
    
    # Get scores for the given user
    item_cf_scores = svd_predictions.loc[user_id].sort_values(ascending=False)
    item_cf_recommendations = pd.DataFrame({'title': item_cf_scores.index, 'score': item_cf_scores.values})
    
    # Normalize scores to range [0, 1]
    item_cf_recommendations['score'] = (item_cf_recommendations['score'] - item_cf_recommendations['score'].min()) / \
                                       (item_cf_recommendations['score'].max() - item_cf_recommendations['score'].min())
    
    # Sample recommendations based on temperature
    item_cf_recommendations = item_cf_recommendations.sample(
        n=n, 
        weights=(item_cf_recommendations['score'] ** (1 / temperature))
    )
    
    return item_cf_recommendations

def jaccard_similarity(set1, set2):
    """
    Computes the Jaccard similarity between two sets.
    
    Parameters:
    set1 (set): First set of elements.
    set2 (set): Second set of elements.
    
    Returns:
    float: Jaccard similarity score between 0 and 1.
    """
    if not set1 or not set2:  # Handle empty sets
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_content_recommendations(df, user_id, user_item_matrix, weights, k=10, temperature=1.0):
    """
    Recommends movies based on a user's past ratings using content-based filtering.
    Integrates actor, genre, and director similarity.
    Samples recommendations based on temperature.
    Saves the recommendations in a CSV file named after the user ID.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing movie data.
    user_id (int): User ID for whom recommendations are to be generated.
    user_item_matrix (pd.DataFrame): User-item matrix with user ratings.
    weights (list): List of weights for actor, genre, and director scores.
    k (int): Number of recommendations to return.
    temperature (float): Temperature parameter for sampling.
    
    Returns:
    pd.DataFrame: DataFrame containing sampled k recommended movies.
    """
    start_time = time.time()  # Start timing the function
    if user_id not in user_item_matrix.index:
        raise ValueError("User ID not found in the dataset.")
    
    # Get the user's rated movies and their ratings
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 3.5].index.tolist()
    rated_movies_with_scores = user_ratings[user_ratings > 3].to_dict()
    
    if not rated_movies:
        return pd.DataFrame()  # Return empty DataFrame if no movies are rated
    
    print(f"[DEBUG] Time to fetch rated movies: {time.time() - start_time:.2f} seconds")
    
    # Apply demographic filtering first to reduce the size of the DataFrame
    demographic_start = time.time()
    df = demographic_filtering(df)
    print(f"[DEBUG] Time to apply demographic filtering: {time.time() - demographic_start:.2f} seconds")
    
    # Filter out movies with zero demographic scores
    df = df[df['dmg_score'] > 0]
    print(f"[DEBUG] Number of movies after demographic filtering: {len(df)}")
    
    # Update compute_similarity to accept x as a parameter
    def compute_similarity(column, x, movie, rating):
        if movie in df['title'].values:
            target_row = df.loc[df['title'] == movie]
            if not target_row.empty:
                if column == 'director':
                    return 1 if x == target_row[column].iloc[0] else 0
                else:
                    target_set = set(target_row[column].iloc[0])
                    return jaccard_similarity(set(x), target_set) * (rating / 5.0)
        return 0.0  # Default value if movie is not found

    # Compute similarity scores for all movies at once
    similarity_start = time.time()
    df['actor_score'] = df['cast'].apply(lambda x: sum(compute_similarity('cast', x, movie, rating) for movie, rating in rated_movies_with_scores.items()))
    print(f"[DEBUG] Time to compute actor scores: {time.time() - similarity_start:.2f} seconds")
    
    similarity_start = time.time()
    df['genre_score'] = df['genres'].apply(lambda x: sum(compute_similarity('genres', x, movie, rating) for movie, rating in rated_movies_with_scores.items()))
    print(f"[DEBUG] Time to compute genre scores: {time.time() - similarity_start:.2f} seconds")
    
    similarity_start = time.time()
    df['diro_score'] = df['director'].apply(lambda x: sum(compute_similarity('director', x, movie, rating) for movie, rating in rated_movies_with_scores.items()))
    print(f"[DEBUG] Time to compute director scores: {time.time() - similarity_start:.2f} seconds")
    
    # Normalize scores
    normalization_start = time.time()
    for col in ['actor_score', 'genre_score', 'diro_score', 'dmg_score']:
        norm_col = f"norm_{col}"
        df[norm_col] = normalize_scores(df[col])
    print(f"[DEBUG] Time to normalize scores: {time.time() - normalization_start:.2f} seconds")
    
    # Calculate weighted score for each movie
    weighted_start = time.time()
    df['weighted_score'] = df.apply(
        lambda row: weighted_score(
            [row['norm_actor_score'], row['norm_genre_score'], row['norm_diro_score'], row['norm_dmg_score']],
            weights
        ), axis=1
    )
    print(f"[DEBUG] Time to calculate weighted scores: {time.time() - weighted_start:.2f} seconds")
    
    # Exclude already rated movies
    exclusion_start = time.time()
    recommendations = df[~df['title'].isin(rated_movies)]
    print(f"[DEBUG] Time to exclude rated movies: {time.time() - exclusion_start:.2f} seconds")
    
    # Sample recommendations based on temperature
    sampling_start = time.time()
    recommendations = recommendations.sample(
        n=k, 
        weights=(recommendations['weighted_score'] ** (1 / temperature))
    )
    print(f"[DEBUG] Time to sample recommendations: {time.time() - sampling_start:.2f} seconds")
    
    # Save the recommendations to a CSV file named after the user ID
    save_start = time.time()
    output_file = f"data/user_{user_id}_recommendations.csv"
    recommendations.to_csv(output_file, index=False)
    print(f"[DEBUG] Time to save recommendations: {time.time() - save_start:.2f} seconds")
    
    print(f"[DEBUG] Total time for get_content_recommendations: {time.time() - start_time:.2f} seconds")
    return recommendations
