import pandas as pd


def load_datasets():
    df_movies = pd.read_csv("data/ml_latest_small/movies.csv")
    df_movies = df_movies.rename(columns={'movieId': 'movie_id'})

    df_ratings = pd.read_csv("data/ml_latest_small/ratings.csv")
    df_ratings = df_ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})

    df_links = pd.read_csv("data/ml_latest_small/links.csv")
    df_links = df_links.rename(columns={'movieId': 'movie_id', 'tmdbId': 'tmdb_id'})[['movie_id', 'tmdb_id']]

    return df_movies, df_ratings, df_links