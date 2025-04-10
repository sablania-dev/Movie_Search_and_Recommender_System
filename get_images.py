import requests
from PIL import Image
from io import BytesIO
import os
import time
import pandas as pd
from tqdm import tqdm  # Add tqdm for progress bar

def get_all_tmdb_ids():
    """
    Extracts a list of all tmdbId values from the links dataset.
    """
    try:
        data = pd.read_csv('data/movies_metadata.csv')
        C= data['vote_average'].mean()
        m= data['vote_count'].quantile(0.9)
        q_movies = data.copy().loc[data['vote_count'] >= m]
        # Convert tmdbId to numeric and drop NaN values
        q_movies['id'] = pd.to_numeric(q_movies['id'], errors='coerce')
        return q_movies['id'].dropna().astype(int).tolist()
    except Exception as e:
        raise RuntimeError(f"Error extracting tmdbId values: {e}")

def fetch_and_save_images(list_of_tmdb_ids, height, width):
    api_key = "b750cc3757f442fddb56342097791eca"
    current_directory = os.getcwd()
    images_folder = os.path.join(current_directory, "images")
    os.makedirs(images_folder, exist_ok=True)
    image_paths = []

    # Wrap the list of tmdb_ids with tqdm for progress tracking
    for tmdb_id in tqdm(list_of_tmdb_ids, desc="Fetching images"):
        save_path = os.path.join(images_folder, f"{tmdb_id}.jpg")
        
        # Check if the image already exists
        if os.path.exists(save_path):
            image_paths.append(save_path)
            continue

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                poster_url = f"http://image.tmdb.org/t/p/w{width}{poster_path}"
                img = requests.get(poster_url)
                img.raise_for_status()
                image = Image.open(BytesIO(img.content))
                image = image.resize((width, height))  # Resize the image
                image.save(save_path)
                image_paths.append(save_path)
            else:
                print(f"Poster not found for TMDB ID: {tmdb_id}")
        except Exception as e:
            print(f"Error fetching image for TMDB ID {tmdb_id}: {e}")
            time.sleep(0.03)

    return image_paths

if __name__ == "__main__":
    # Get all tmdbId values
    list_of_tmdb_ids = get_all_tmdb_ids()
    # list_of_tmdb_ids = [862]
    
    # Define desired image dimensions
    height, width = 500, 500
    
    # Fetch and save images
    image_paths = fetch_and_save_images(list_of_tmdb_ids, height, width)
    print(f"Images saved at: {image_paths}")