import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id','title','overview','genres','keywords']]

# Drop missing values
movies.dropna(inplace=True)

# Combine columns to create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']

# Vectorize the text
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return ["Movie not found"]
    
    idx = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended