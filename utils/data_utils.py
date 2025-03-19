import re


def get_unique_movie_genre(df_movies):
    unique_movie_genre = sorted(list(set('|'.join(df_movies.genres).split('|'))))
    return unique_movie_genre


def extract_year(title: str) -> int:
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

def split_movie_genres(df_movies):
    # Use str.get_dummies() to create binary columns for each genre
    df_genres = df_movies['genres'].str.get_dummies(sep='|')

    # Drop the original genres column and merge the new one-hot encoded genres
    df_movies_split_genre = df_movies.drop(columns=['genres']).join(df_genres)
    return df_movies_split_genre


def preprocess_df_movie(df_movies, unique_movie_genre):
    df_movie_preprocessed = split_movie_genres(df_movies)
    df_movie_preprocessed.loc[:, 'year'] = df_movie_preprocessed.title.apply(extract_year)
    df_movie_preprocessed = df_movie_preprocessed[['movie_id', 'year'] + unique_movie_genre]
    return df_movie_preprocessed


def calculate_split_timestamp(df_ratings, test_size):
    return df_ratings.timestamp.quantile(q=1 - test_size)


def split_train_test(dataset, split_timestamp):
    dataset = dataset.sort_values(['timestamp', 'user_id', 'movie_id'])

    # Train-test split
    dataset_train = dataset[dataset['timestamp'] < split_timestamp].reset_index(drop=True)
    dataset_test = dataset[dataset['timestamp'] >= split_timestamp].reset_index(drop=True)
    return dataset_train, dataset_test


def build_df_true_positives(df_ratings, split_timestamp):
    df_true_positives = (
        df_ratings[df_ratings.timestamp >= split_timestamp]
        .groupby('user_id', as_index=False)['movie_id']
        .apply(lambda x: list(x))
    )

    df_true_positives.columns = ['user_id', 'true_positives']
    return df_true_positives
