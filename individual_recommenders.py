import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from datetime import datetime, timedelta

#### DEMOGRAPHIC FILTERING ####

def demographic_filtering(df, quantile=0.9):
    """
    Applies demographic filtering to the DataFrame.
    Adds a 'dmg_score' column based on the IMDB weighted rating formula.
    Sets dmg_score to zero for non-qualifying movies.
    Updates the preprocessed CSV file if 'dmg_score' is not already present.
    Returns the updated DataFrame.
    """
    preprocessed_file = 'data/preprocessed_movies.csv'

    # Check if the preprocessed file contains the 'dmg_score' column
    if os.path.exists(preprocessed_file):
        preprocessed_df = pd.read_csv(preprocessed_file)
        if 'dmg_score' in preprocessed_df.columns:
            df['dmg_score'] = preprocessed_df['dmg_score']
            return df

    # Compute 'dmg_score' if not already present
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(quantile)
    df = df.copy()
    
    # Calculate the IMDB weighted rating formula
    df['dmg_score'] = df.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + 
                  (m / (m + x['vote_count']) * C) if x['vote_count'] >= m else 0, axis=1
    )
    
    # Ensure 'dmg_score' is numeric after computation
    df['dmg_score'] = pd.to_numeric(df['dmg_score'], errors='coerce').fillna(0)
    
    # Update the preprocessed CSV file with the new 'dmg_score' column
    if os.path.exists(preprocessed_file):
        preprocessed_df = pd.read_csv(preprocessed_file)
        preprocessed_df['dmg_score'] = df['dmg_score']
        preprocessed_df.to_csv(preprocessed_file, index=False)
    else:
        df.to_csv(preprocessed_file, index=False)
    
    return df

#### PLOT BASED ####

#### COLLABORATIVE FILTERING ####

def svd_recommender(user_item_matrix, num_components=20):
    """
    Applies Singular Value Decomposition (SVD) for collaborative filtering.
    Returns predicted ratings matrix.
    """
    svd = TruncatedSVD(n_components=num_components)
    svd_matrix = svd.fit_transform(user_item_matrix)
    reconstructed_matrix = svd_matrix @ svd.components_
    return pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

def cosine_recommender(user_item_matrix):
    """
    Computes cosine similarity between users and recommends movies based on similar users.
    Returns similarity matrix.
    """
    similarity_matrix = cosine_similarity(user_item_matrix)
    return pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

#### TRENDING / POPULARITY-BASED RECOMMENDERS ####

def popular_right_now(df, temperature=1.0, n=10):
    """
    Recommends movies that are popular right now based on popularity and recent release dates.
    """
    # Calculate the current date as the latest release_date in the dataset
    current_date = pd.to_datetime(df['release_date'], errors='coerce').max()
    six_months_ago = current_date - timedelta(days=180)

    # Filter movies released in the past 6 months
    recent_movies = df[pd.to_datetime(df['release_date'], errors='coerce') >= six_months_ago]

    # Adjust sample size if the population is smaller than n
    sample_size = min(len(recent_movies), n)

    # Sample recommendations based on popularity
    if sample_size > 0:
        recommendations = recent_movies.sample(
            n=sample_size,
            weights=(recent_movies['popularity'] ** (1 / temperature))
        )
        return recommendations
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no movies match the criteria


def top_grossing(df, temperature=1.0, n=10):
    """
    Recommends top-grossing movies based on revenue.
    """
    # Sample recommendations based on revenue
    recommendations = df.sample(
        n=n,
        weights=(df['revenue'] ** (1 / temperature))
    )
    return recommendations


def critically_acclaimed(df, temperature=1.0, n=10):
    """
    Recommends critically acclaimed movies based on dmg_score.
    """
    # Sample recommendations based on dmg_score
    recommendations = df.sample(
        n=n,
        weights=(df['dmg_score'] ** (1 / temperature))
    )
    return recommendations


def hidden_gems(df, temperature=1.0, n=10):
    """
    Recommends hidden gems: movies with low popularity but high vote_average or dmg_score.
    """
    # Filter movies with low popularity and high vote_average or dmg_score
    hidden_gems_df = df[(df['popularity'] < df['popularity'].quantile(0.3)) &
                        (df['vote_average'] > df['vote_average'].quantile(0.7))]

    # Filter rows with non-zero weights
    hidden_gems_df = hidden_gems_df[hidden_gems_df['dmg_score'] > 0]

    # Adjust sample size if the population is smaller than n
    sample_size = min(len(hidden_gems_df), n)

    # Sample recommendations based on dmg_score
    if sample_size > 0:
        recommendations = hidden_gems_df.sample(
            n=sample_size,
            weights=(hidden_gems_df['dmg_score'] ** (1 / temperature))
        )
        return recommendations
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no movies match the criteria