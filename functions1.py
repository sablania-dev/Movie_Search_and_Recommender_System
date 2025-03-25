def weighted_score(scores, weights):
    """
    Calculates the weighted score given a list of scores and corresponding weights.
    
    :param scores: List of scores (each between 0 and 1).
    :param weights: List of weights (each between 0 and 1).
    :return: Weighted score (between 0 and 1).
    """
    if len(scores) != len(weights):
        raise ValueError("Scores and weights must have the same number of elements.")

    if not (all(0 <= s <= 1 for s in scores) and all(0 <= w <= 1 for w in weights)):
        raise ValueError("All scores and weights must be between 0 and 1.")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0  # Avoid division by zero, return a neutral score

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight

# Example usage
scores = [0.8, 0.7, 0.9, 0.6, 0.5]
weights = [0.2, 0.1, 0.3, 0.25, 0.15]

print(weighted_score(scores, weights))  # Output: weighted average score

import difflib
import pandas as pd

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

# Example usage:
data = {'title': ['Titanic', 'Avatar', 'Inception', 'Interstellar']}
df = pd.DataFrame(data)
print(autocomplete(df, 'titan'))  # Output: 'Titanic'