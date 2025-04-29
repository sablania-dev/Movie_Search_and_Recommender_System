import os
import pandas as pd
import numpy as np
from individual_recommenders import demographic_filtering  # Import the function


def load_data():
    try:
        df1 = pd.read_csv('data/tmdb_5000_credits.csv')
        df2 = pd.read_csv('data/tmdb_5000_movies.csv')

        # Let's join the two dataset on the 'id' column
        df1.columns = ['id', 'tittle', 'cast', 'crew']
        df2 = df2.merge(df1, on='id')

        # drop column tittle from df2
        df2 = df2.drop('tittle', axis=1)

        return df2
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# Parse the stringified features into their corresponding python objects
from ast import literal_eval
      
# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def preprocess_data(df):
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    # Define new director, cast, genres and keywords features that are in a suitable form.
    df['director'] = df['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(get_list)

    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        df[feature] = df[feature].apply(clean_data)
    
    # Apply demographic filtering to add the 'dmg_score' column
    df = demographic_filtering(df)

    return df

def load_collaborative_data():
    """
    Loads the raw data required for collaborative filtering.
    Returns ratings, links, and movies_metadata DataFrames.
    """
    try:
        ratings = pd.read_csv('data/ratings.csv')
        links = pd.read_csv('data/links.csv')
        movies_metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)
        return ratings, links, movies_metadata
    except Exception as e:
        raise RuntimeError(f"Error loading collaborative data: {e}")

def preprocess_collaborative_data(ratings, links, movies_metadata):
    """
    Prepares the user-item matrix for collaborative filtering.
    """
    # Convert ID columns to consistent types
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
    
    # Merge ratings with links to get tmdbId
    ratings = ratings.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner')
    
    # Merge with movies_metadata to get movie titles
    ratings = ratings.merge(movies_metadata[['id', 'title']], left_on='tmdbId', right_on='id', how='inner')
    
    # Count user ratings and filter experienced users (rated ≥200 movies)
    user_counts = ratings['userId'].value_counts()
    experienced_users = user_counts[user_counts >= 200].index
    ratings = ratings[ratings['userId'].isin(experienced_users)]
    
    # Count movie ratings and filter popular movies (rated by ≥50 users)
    movie_counts = ratings['title'].value_counts()
    popular_movies = movie_counts[movie_counts >= 50].index
    ratings = ratings[ratings['title'].isin(popular_movies)]
    
    # Pivot into user-item matrix
    user_item_matrix = ratings.pivot_table(index='userId', columns='title', values='rating', fill_value=0)
    
    return user_item_matrix

def load_or_preprocess_data():
    """
    Checks if preprocessed data exists. If not, preprocesses and saves the data.
    """
    preprocessed_file = 'data/preprocessed_movies.csv'
    if os.path.exists(preprocessed_file):
        return pd.read_csv(preprocessed_file)
    else:
        raw_data = load_data()
        preprocessed_data = preprocess_data(raw_data)
        preprocessed_data.to_csv(preprocessed_file, index=False)
        return preprocessed_data

def load_or_preprocess_collaborative_data():
    """
    Checks if preprocessed collaborative data exists in CSV format.
    If not, preprocesses and saves the data in CSV format.
    """
    preprocessed_file = 'data/preprocessed_user_item_matrix.csv'
    if os.path.exists(preprocessed_file):
        # Load preprocessed collaborative data in CSV format
        user_item_matrix = pd.read_csv(preprocessed_file, index_col=0)
        return user_item_matrix
    else:
        # Preprocess and save collaborative data
        ratings, links, movies_metadata = load_collaborative_data()
        user_item_matrix = preprocess_collaborative_data(ratings, links, movies_metadata)
        user_item_matrix.to_csv(preprocessed_file)
        return user_item_matrix
