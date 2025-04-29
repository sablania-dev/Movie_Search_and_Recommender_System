import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Import tqdm for progress bar

# Load preprocessed_movies.csv
preprocessed_file = 'data/preprocessed_movies.csv'
df = pd.read_csv(preprocessed_file)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute sentence embeddings for the 'overview' column with a progress bar
tqdm.pandas(desc="Computing Embeddings")
df['plot_embedding'] = df['overview'].fillna("").progress_apply(lambda x: model.encode(x))

# Get the embedding for movie ID 41154
target_movie_id = 41154
target_embedding = df.loc[df['id'] == target_movie_id, 'plot_embedding'].values[0]

# Compute cosine similarity between the target movie and all other movies with a progress bar
tqdm.pandas(desc="Computing Cosine Similarity")
df['cosine_similarity'] = df['plot_embedding'].progress_apply(lambda x: cosine_similarity([target_embedding], [x])[0][0])

# Get the top 10 similar movies
top_similar_movies = df.nlargest(10, 'cosine_similarity')[['id', 'title', 'cosine_similarity']]

# Print the top 10 similar movies
print(top_similar_movies)
