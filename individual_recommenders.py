import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

#### CONTENT-BASED FILTERING ####
#### ACTOR, DIRECTOR, KEYWORDS, GENRES ####

# Function to compute Jaccard similarity
def jaccard_similarity(set1, set2):
    if not set1 or not set2:  # Handle empty sets
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Function to compute similarity scores for a given movie title
def actors_director_keywords_genres(df, title):
    if title not in df['title'].values:
        return "Movie not found in dataset."

    # Get the target movie row
    target_movie = df[df['title'] == title].iloc[0]

    # Convert list-like columns to sets
    target_cast = set(target_movie['cast'])
    target_keywords = set(target_movie['keywords'])
    target_genres = set(target_movie['genres'])
    target_director = target_movie['director']

    # Compute Jaccard similarity scores for all movies
    df['actor_score'] = df['cast'].apply(lambda x: jaccard_similarity(set(x), target_cast))
    df['genre_score'] = df['genres'].apply(lambda x: jaccard_similarity(set(x), target_genres))
    df['kwd_score'] = df['keywords'].apply(lambda x: jaccard_similarity(set(x), target_keywords))

    # Director match (Binary Score: 1 if same, 0 otherwise)
    df['diro_score'] = df['director'].apply(lambda x: 1 if x == target_director else 0)

    return df


#### DEMOGRAPHIC FILTERING ####

def demographic_filtering(df, quantile=0.9):
    """
    Applies demographic filtering to the DataFrame.
    Adds a 'dmg_score' column based on the IMDB weighted rating formula.
    Returns the updated DataFrame.
    """
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(quantile)
    df = df.copy()
    df['dmg_score'] = df.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + 
                  (m / (m + x['vote_count']) * C), axis=1
    )
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