import difflib
import pandas as pd
from individual_recommenders import actors_director_keywords_genres

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


def get_k_recommendations(df: pd.DataFrame, title: str, k: int) -> pd.DataFrame:
    """
    Get top k movie recommendations based on similarity scores.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing movie data with similarity scores.
    title (str): The title of the movie for which recommendations are to be found.
    k (int): The number of recommendations to return.
    
    Returns:
    pd.DataFrame: DataFrame containing top k recommended movies.
    """
    if title not in df['title'].values:
        raise ValueError("Movie not found in dataset.")
    
    # Compute similarity scores
    df = actors_director_keywords_genres(df, title)
    
    # Ensure required columns exist and are numeric
    for col in ['actor_score', 'genre_score', 'kwd_score', 'diro_score']:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is missing or not numeric in the DataFrame.")
    
    # Calculate weighted score for each row
    df['weighted_score'] = df.apply(
        lambda row: weighted_score(
            [row['actor_score'], row['genre_score'], row['kwd_score'], row['diro_score']],
            [0.25, 0.25, 0.25, 0.25]
        ), axis=1
    )
    
    # Sort by weighted score and return top k recommendations
    recommendations = df.sort_values(by='weighted_score', ascending=False).head(k)
    
    return recommendations[['title', 'weighted_score']]

