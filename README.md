# Movie Search and Recommender System

## 1. Dataset
For our movie recommender system, we have selected the following datasets and intend to join them:

- **The Movies Dataset**: Contains movie metadata, user ratings, and additional attributes necessary for recommendation.
- **TMDB Movie Metadata**: Provides extensive metadata such as genres, cast, keywords, and descriptions, enabling content-based recommendations.

## 2. Problem Statement
The goal of this project is to build a movie recommender system that enhances user experience by suggesting relevant movies. Instead of using a single recommendation technique, we aim to implement multiple filtering methods that users can configure based on their preferences. The system will:

- Provide personalized movie recommendations based on past user ratings.
- Allow users to search for movies with semantic similarity, ensuring relevant results even with partial or vague queries.
- Enable dynamic weighting of different recommendation strategies to improve flexibility.

## 3. Proposed Approach & Algorithm

### Website Development
The recommender system will be integrated into a web application where users can log in, search for movies, and receive personalized recommendations.

- The frontend will be built using **React**, while the backend will be developed using **Flask/Django** to handle API requests and data processing.
- User authentication will be implemented to store and track user preferences, ratings, and search history.

### Semantic Search Feature
- The search functionality will leverage **semantic similarity**, enabling users to find movies based on contextual relevance rather than exact matches.
- **Word embeddings** (e.g., **BERT, Sentence Transformers**) will be used to generate vector representations of movie titles, descriptions, and keywords.
- **Cosine similarity** or **nearest neighbor search** will be applied to rank and return relevant search results.

### Recommender System

#### Weighted Rating Method
- Implements **IMDb-like weighted scoring** based on movie popularity and ratings.

#### Content-Based Filtering
- Uses **TF-IDF** and **cosine similarity** for textual features like genres, keywords, and plot descriptions.
- Employs **word embeddings** (e.g., **BERT, Sentence Transformers**) to capture deeper semantic similarity.

#### Collaborative Filtering
- **User-user** and **item-item collaborative filtering** based on rating patterns.
- **Matrix factorization** techniques like **Singular Value Decomposition (SVD)** or **Alternating Least Squares (ALS)** for latent factor analysis.

#### Hybrid Model & Configurable Weights
- Combines the above methods in a **weighted ensemble approach**.
- Allows users to **adjust weights dynamically** through the web interface to fine-tune recommendations.

## 4. Backend Functions

### Function 1: Individual Recommendation Scores
```python
get_individual_recommendation_score_for_all_movies(DataFrame df, string X)
```
**Returns:**
- `DataFrame individual_recommendation_scores` (with an added column `'xyz_score'` ranging from 0 to 1)
- Computes individual recommendation scores from different recommender systems.
- Each function provides a score `s_i`, and the final recommendation score is:
  
  \[ R = \sum (s_i \cdot w_i), \quad \sum w_i = 1 \]

### Function 2: Total Weighted Recommendation Scores
```python
get_recommendation_score_for_all_movies(DataFrame df, string X, list custom_weights)
```
**Returns:**
- `DataFrame final_recommendation_scores` (with an added column `'final_score'` ranging from 0 to 1)

### Function 3: Top k Recommendations
```python
get_k_recommendations(DataFrame df, string X, int k)
```
**Returns:**
- `list top_k_movie_recommendations`

### Function 4: Autocomplete
```python
autocomplete(DataFrame df, string X)
```
**Returns:**
- `string nearest_keyword_match`
- Example: Input = `'titan'` → Output = `'Titanic'`

### Function 5: Collaborative Filtering
```python
get_collab_recommendation_score_for_all_movies(DataFrame df, string user_id)
```
**Returns:**
- `DataFrame collab_recommendation_scores` (with an added column `'collab_score'` ranging from 0 to 1)
- Provides recommendations based on **user's rating history**.

## 5. List of Individual Recommender Systems

| Filtering Type         | Input Column(s)        | Output Column  | Miscellaneous |
|------------------------|------------------------|----------------|---------------|
| **Demographic Filtering** | `vote_count`, `vote_average` | `dmg_score` | `m = minimum votes required` |
| **Popular Movies**     | `popularity`          | None           | Direct and straightforward |
| **Overview**          | `overview`            | `ovrw_score`   | **Step 1**: Vector embeddings using `'all-MiniLM-L6-v2'` <br> **Step 2**: Cosine similarity (FAISS) |
| **Director**          | `crew`                | `diro_score`   | Binary score |
| **Genre**            | `genre`               | `genre_score`  | Jaccard Similarity |
| **Actors**           | `cast`                | `actor_score`  | - |
| **Keywords**         | `keywords`            | `kwd_score`    | - |
| **Collaborative Filtering** | - | `collab_score` | Based on user rating history |
