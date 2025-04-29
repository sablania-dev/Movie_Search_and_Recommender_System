import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import os

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