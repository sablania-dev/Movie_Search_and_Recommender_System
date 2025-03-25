import pandas as pd

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

#### PLOT BASED ####

#### COLLABORATIVE FILTERING ####