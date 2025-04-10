import difflib
import pandas as pd
from individual_recommenders import actors_director_keywords_genres
from loading_and_preprocessing import preprocess_collaborative_data
from individual_recommenders import svd_recommender, cosine_recommender, demographic_filtering
import os
from PIL import Image
import streamlit as st

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
    
    return recommendations[['title', 'weighted_score']]

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
    Shows movie metadata on the right side of each image.
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
            if 'vote_average' in row:
                st.write(f"**Rating:** {row['vote_average']}")
            if 'vote_count' in row:
                st.write(f"**Votes:** {row['vote_count']}")
            if 'genres' in row:
                st.write(f"**Genres:** {row['genres']}")
            if 'director' in row:
                st.write(f"**Director:** {row['director']}")
