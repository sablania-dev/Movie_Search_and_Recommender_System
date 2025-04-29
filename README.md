# Movie Search and Recommender System

## Overview
This project is a **Movie Search and Recommender System** designed to enhance user experience by providing personalized movie recommendations and enabling semantic search functionality. The system integrates multiple recommendation techniques, allowing users to dynamically adjust weights for different strategies to fine-tune their recommendations.

## Features
- **Personalized Recommendations**: Suggests movies based on user preferences, past ratings, and search history.
- **Semantic Search**: Enables users to search for movies using contextual relevance rather than exact matches.
- **Dynamic Weight Adjustment**: Allows users to configure weights for different recommendation strategies through an intuitive interface.
- **Hybrid Recommendation Model**: Combines content-based, collaborative filtering, and popularity-based methods for better accuracy.
- **Interactive Web Interface**: Built using **Streamlit**, providing a user-friendly experience.

## Dataset
The system uses the following datasets:
- **The Movies Dataset**: Contains metadata, user ratings, and additional attributes for movies.
- **TMDB Movie Metadata**: Provides extensive metadata such as genres, cast, keywords, and descriptions.

## Recommendation Techniques
### 1. **Content-Based Filtering**
- Uses **TF-IDF** and **cosine similarity** for textual features like genres, keywords, and plot descriptions.
- Employs **word embeddings** (e.g., **BERT, Sentence Transformers**) to capture deeper semantic similarity.

### 2. **Collaborative Filtering**
- Implements **user-user** and **item-item collaborative filtering** based on rating patterns.
- Uses **Matrix Factorization** techniques like **Singular Value Decomposition (SVD)** for latent factor analysis.

### 3. **Popularity-Based Recommendations**
- Recommends movies based on popularity, revenue, or critical acclaim.

### 4. **Hybrid Model**
- Combines the above methods in a **weighted ensemble approach**.
- Allows users to dynamically adjust weights for different components.

## Individual Recommender Systems
| Filtering Type         | Input Column(s)        | Output Column  | Miscellaneous |
|------------------------|------------------------|----------------|---------------|
| **Demographic Filtering** | `vote_count`, `vote_average` | `dmg_score` | IMDb-like weighted scoring |
| **Popular Movies**     | `popularity`          | None           | Direct and straightforward |
| **Overview**           | `overview`            | `ovrw_score`   | Vector embeddings and cosine similarity |
| **Director**           | `crew`                | `diro_score`   | Binary score |
| **Genre**              | `genre`               | `genre_score`  | Jaccard Similarity |
| **Actors**             | `cast`                | `actor_score`  | Jaccard Similarity |
| **Keywords**           | `keywords`            | `kwd_score`    | Jaccard Similarity |

## Web Application
The recommender system is integrated into a web application where users can:
- Log in with their credentials.
- Search for movies using semantic search.
- Select from various recommendation types:
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
  - Content-Based Filtering
  - Popular Right Now
  - Top Grossing
  - Critically Acclaimed
  - Hidden Gems
- Adjust weights for scoring components dynamically.
- View recommendations with metadata and images.

## Technologies Used
- **Frontend**: Streamlit
- **Backend**: Python (Pandas, NumPy, Scikit-learn)
- **Data Processing**: Preprocessing scripts for cleaning and transforming data.
- **Recommendation Algorithms**: Content-based filtering, collaborative filtering, and hybrid models.
- **Visualization**: Matplotlib, PIL for displaying images.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommender-system.git
   cd movie-recommender-system
   ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Run the application: 
  ```bash
  streamlit run main.py
  ```

4. Open the application in your browser at http://localhost:8501. 
## Folder Structure
├── data/                     # Contains datasets and preprocessed files
├── images/                   # Contains movie images
├── [main.py](http://_vscodecontentref_/0)                   # Main Streamlit application
├── [functions1.py](http://_vscodecontentref_/1)             # Core functions for recommendations
├── [individual_recommenders.py](http://_vscodecontentref_/2) # Individual recommendation algorithms
├── [loading_and_preprocessing.py](http://_vscodecontentref_/3) # Data loading and preprocessing scripts
├── [README.md](http://_vscodecontentref_/4)                 # Project documentation
├── [requirements.txt](http://_vscodecontentref_/5)          # Python dependencies 
## Future Enhancements
Add user authentication with a database.
Improve semantic search using advanced NLP models.
Integrate real-time data updates from external APIs.
Deploy the application on a cloud platform for public access.