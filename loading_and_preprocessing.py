import pandas as pd
import numpy as np


def load_data(which_data=0):
    
    if which_data == 0:
        df1 = pd.read_csv('data/tmdb_5000_credits.csv')
        df2 = pd.read_csv('data/tmdb_5000_movies.csv')

        # Let's join the two dataset on the 'id' column
        df1.columns = ['id', 'tittle', 'cast', 'crew']
        df2 = df2.merge(df1, on='id')

        # drop column tittle from df2
        df2 = df2.drop('tittle', axis=1)

        return df2

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