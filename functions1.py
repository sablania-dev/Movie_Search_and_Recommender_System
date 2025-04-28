import difflib
import pandas as pd
from individual_recommenders import actors_director_keywords_genres
from loading_and_preprocessing import preprocess_collaborative_data
from individual_recommenders import svd_recommender, cosine_recommender, demographic_filtering, jaccard_similarity
import os
from PIL import Image
import streamlit as st
import numpy as np

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


def autocomplete(df: pd.DataFrame, X: str) -> str:
    """
    Returns the nearest keyword match from df['title'] based on the input string X.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing a 'title' column.
    X (str): Input string to be matched.
    
    Returns:
    str: The closest matching title from the DataFrame.
    """
    if 'title' not in df.columns:
        raise ValueError("DataFrame must contain a 'title' column")
    
    titles = df['title'].astype(str).tolist()
    matches = difflib.get_close_matches(X, titles, n=1, cutoff=0.4)
    
    return matches[0] if matches else "No match found"


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

def get_k_recommendations(df: pd.DataFrame, title: str, k: int, weights=None) -> pd.DataFrame:
    """
    Get top k movie recommendations based on similarity scores and demographic score.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing movie data with similarity scores.
    title (str): The title of the movie for which recommendations are to be found.
    k (int): The number of recommendations to return.
    weights (list): List of weights for actor, genre, keywords, director, and demographic scores.
    
    Returns:
    pd.DataFrame: DataFrame containing top k recommended movies.
    """
    if title not in df['title'].values:
        raise ValueError("Movie not found in dataset.")
    
    # Default weights if none are provided
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Compute similarity scores
    df = actors_director_keywords_genres(df, title)
    df = demographic_filtering(df)
    
    # Ensure required columns exist and are numeric
    for col in ['actor_score', 'genre_score', 'kwd_score', 'diro_score', 'dmg_score']:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is missing or not numeric in the DataFrame.")
    
    # Normalize scores using the new normalization function and add new columns
    for col in ['actor_score', 'genre_score', 'kwd_score', 'diro_score', 'dmg_score']:
        norm_col = f"norm_{col}"
        df[norm_col] = normalize_scores(df[col])
    
    # Calculate weighted score for each row, incorporating normalized scores
    df['weighted_score'] = df.apply(
        lambda row: weighted_score(
            [row['norm_actor_score'], row['norm_genre_score'], row['norm_kwd_score'], row['norm_diro_score'], row['norm_dmg_score']],
            weights
        ), axis=1
    )
    
    # Sort by weighted score and return top k recommendations
    recommendations = df.sort_values(by='weighted_score', ascending=False).head(k)
    
    return recommendations

def get_collab_recommendation_score_for_all_movies(user_item_matrix, user_id, svd_predictions):
    """
    Provides collaborative filtering recommendations based on a user's rating history.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError("User ID not found in the dataset.")
    
    # Use precomputed SVD predictions
    user_svd_scores = svd_predictions.loc[user_id]
    
    # Compute user-user similarity using cosine similarity
    user_similarity = cosine_recommender(user_item_matrix)
    similar_users = user_similarity[user_id].drop(user_id)
    
    # Weighted collaborative score (blending SVD and user similarity)
    collab_scores = (user_svd_scores * 0.7) + (similar_users.mean() * 0.3)
    
    # Normalize scores to range 0-1
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())
    
    return pd.DataFrame({'title': collab_scores.index, 'collab_score': collab_scores.values}).sort_values(by='collab_score', ascending=False)

def display_results_with_images(results_df):
    """
    Displays results horizontally with images and titles.
    Shows movie metadata on the right side of each image, including weighted score.
    If an image does not exist, a blank template is shown.
    """
    current_directory = os.getcwd()
    images_folder = os.path.join(current_directory, "images")
    
    for index, row in results_df.iterrows():
        # Use the index as the movie ID
        movie_id = index
        title = row['title']
        image_path = os.path.join(images_folder, f"{movie_id}.jpg")
        
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

def get_content_based_recommendations(df, user_id, user_item_matrix, weights, k=10, temperature=1.0):
    """
    Recommends movies based on a user's past ratings using content-based filtering.
    Handles all movies at once and computes top recommendations for a user.
    Samples recommendations based on temperature.
    Saves the recommendations in a CSV file named after the user ID.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing movie data.
    user_id (int): User ID for whom recommendations are to be generated.
    user_item_matrix (pd.DataFrame): User-item matrix with user ratings.
    weights (list): List of weights for actor, genre, keywords, director, and demographic scores.
    k (int): Number of recommendations to return.
    temperature (float): Temperature parameter for sampling.
    
    Returns:
    pd.DataFrame: DataFrame containing sampled k recommended movies.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError("User ID not found in the dataset.")
    
    # Get the user's rated movies and their ratings
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index.tolist()
    rated_movies_with_scores = user_ratings[user_ratings > 0].to_dict()
    
    if not rated_movies:
        return pd.DataFrame()  # Return empty DataFrame if no movies are rated
    
    # Helper function to compute similarity safely
    def compute_similarity(column, movie, rating):
        if movie in df['title'].values:
            target_row = df.loc[df['title'] == movie]
            if not target_row.empty:
                target_set = set(target_row[column].iloc[0])
                return jaccard_similarity(set(column), target_set) * (rating / 5.0)
        return 0.0  # Default value if movie is not found

    # Compute similarity scores for all movies at once
    df['actor_score'] = df['cast'].apply(lambda x: sum(compute_similarity('cast', movie, rating) for movie, rating in rated_movies_with_scores.items()))
    df['genre_score'] = df['genres'].apply(lambda x: sum(compute_similarity('genres', movie, rating) for movie, rating in rated_movies_with_scores.items()))
    df['kwd_score'] = df['keywords'].apply(lambda x: sum(compute_similarity('keywords', movie, rating) for movie, rating in rated_movies_with_scores.items()))
    df['diro_score'] = df['director'].apply(lambda x: sum((1 if x == df.loc[df['title'] == movie, 'director'].iloc[0] else 0) * (rating / 5.0) for movie, rating in rated_movies_with_scores.items() if movie in df['title'].values))
    
    # Apply demographic filtering
    df = demographic_filtering(df)
    
    # Normalize scores
    for col in ['actor_score', 'genre_score', 'kwd_score', 'diro_score', 'dmg_score']:
        norm_col = f"norm_{col}"
        df[norm_col] = normalize_scores(df[col])
    
    # Calculate weighted score for each movie
    df['weighted_score'] = df.apply(
        lambda row: weighted_score(
            [row['norm_actor_score'], row['norm_genre_score'], row['norm_kwd_score'], row['norm_diro_score'], row['norm_dmg_score']],
            weights
        ), axis=1
    )
    
    # Exclude already rated movies
    recommendations = df[~df['title'].isin(rated_movies)]
    
    # Sample recommendations based on temperature
    recommendations = recommendations.sample(
        n=k, 
        weights=(recommendations['weighted_score'] ** (1 / temperature))
    )
    
    # Save the recommendations to a CSV file named after the user ID
    output_file = f"data/user_{user_id}_recommendations.csv"
    recommendations.to_csv(output_file, index=False)
    
    return recommendations
