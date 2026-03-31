import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ UI ------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Recommend movies by similarity, genre, or rating")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    
    # Include rating column
    movies = movies[['title', 'overview', 'genres', 'keywords', 'vote_average']]

    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    def collapse(L):
        return [i.replace(" ", "") for i in L]

    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)

    movies['overview'] = movies['overview'].fillna('')
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    return movies

movies = load_data()


@st.cache_data
def create_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = create_similarity()


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]


all_genres = set()
for genre_list in movies['genres']:
    for g in genre_list:
        all_genres.add(g)

all_genres = sorted(list(all_genres))


st.subheader("🎥 Recommend by Movie")

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend Similar Movies"):
    recommendations = recommend(selected_movie)
    
    st.subheader("🎯 Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)


st.subheader("🎭 Recommend by Genre")

selected_genre = st.selectbox("Choose a genre", all_genres)

def recommend_by_genre(genre):
    filtered_movies = movies[movies['genres'].apply(lambda x: genre in x)]
    return filtered_movies.sample(10)['title'].values

if st.button("Recommend by Genre"):
    results = recommend_by_genre(selected_genre)
    
    st.subheader(f"🎯 {selected_genre} Movies:")
    for movie in results:
        st.write("👉", movie)


st.subheader("⭐ Top Rated Movies")

min_rating = st.slider("Select minimum rating", 0.0, 10.0, 7.0)

def recommend_by_rating(rating):
    filtered = movies[movies['vote_average'] >= rating]
    return filtered.sort_values(by='vote_average', ascending=False).head(10)

if st.button("Show Top Rated Movies"):
    results = recommend_by_rating(min_rating)
    
    st.subheader(f"🎯 Movies with rating ≥ {min_rating}")
    for i, row in results.iterrows():
        st.write(f"👉 {row['title']} (⭐ {row['vote_average']})")


st.sidebar.title("About")
st.sidebar.write("""
This Movie Recommendation System uses:
- Content-Based Filtering
- CountVectorizer
- Cosine Similarity
- Rating-based filtering

Built using Streamlit.
""")