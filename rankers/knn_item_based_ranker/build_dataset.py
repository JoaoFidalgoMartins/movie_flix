import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.load_datasets import load_datasets
from utils.data_utils import (
    get_unique_movie_genre,
    preprocess_df_movie,
    calculate_split_timestamp,
    split_train_test,
    build_df_true_positives
)
from rankers.knn_item_based_ranker.config import CONFIG


class KNNItemBasedBuildDataset:
    def __init__(self):
        pass

    @staticmethod
    def build_df_movie_features(df_movies):
        unique_movie_genre = get_unique_movie_genre(df_movies)
        df_movies_preprocessed = preprocess_df_movie(df_movies, unique_movie_genre)
        dataset = df_movies_preprocessed.set_index('movie_id')[['year'] + unique_movie_genre]
        return dataset

    @staticmethod
    def preprocess_df_movie_features(df_movie_features):
        dataset_preprocessed = df_movie_features
        preprocessor_year = ColumnTransformer([
            ('fill_na_and_scale', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing 'year' values with mean
                ('scaler', MinMaxScaler())  # MinMax Scaling for 'year' column
            ]), ['year']),
        ], remainder='passthrough')  # Keep all other columns as is

        # Normalize all columns
        # normalizer = Normalizer(norm='l2')

        # Fit and transform training data, then transform test data
        dataset_preprocessed = preprocessor_year.fit_transform(df_movie_features)
        # dataset_preprocessed = normalizer.fit_transform(dataset_preprocessed)

        # Convert back to DataFrame for readability
        dataset_preprocessed = pd.DataFrame(
            dataset_preprocessed,
            columns=df_movie_features.columns,
            index=df_movie_features.index
        )
        return dataset_preprocessed

    @staticmethod
    def build_df_user_last_movie_train(df_ratings):
        split_timestamp = calculate_split_timestamp(df_ratings, CONFIG["test_size"])
        df_ratings_train, df_ratings_test = split_train_test(df_ratings, split_timestamp)
        df_user_last_movie_test = (
            df_ratings_train
            .groupby('user_id', as_index=False)[['timestamp']].max()
            .merge(df_ratings_train, on=['user_id', 'timestamp'], how='inner')
            .groupby('user_id', as_index=False)[['movie_id']].max()
            .reset_index(drop=True)
        )
        df_user_last_movie_test = df_user_last_movie_test[['user_id', 'movie_id']]
        return df_user_last_movie_test

    @staticmethod
    def build_df_user_top_n_last_movie_test(df_ratings, top_n=5):
        split_timestamp = calculate_split_timestamp(df_ratings, CONFIG["test_size"])
        df_ratings_train, df_ratings_test = split_train_test(df_ratings, split_timestamp)

        df_user_top_n_last_movie_test = (
            df_ratings_train
            .sort_values(['user_id', 'timestamp', 'movie_id'], ascending=[True, False, True])
        )
        df_user_top_n_last_movie_test['rank_over_user'] = (
            df_user_top_n_last_movie_test.groupby(['user_id']).cumcount() + 1
        )
        df_user_top_n_last_movie_test = (
            df_user_top_n_last_movie_test[df_user_top_n_last_movie_test.rank_over_user <= 5]
        )
        df_user_top_n_last_movie_test = (
            df_user_top_n_last_movie_test
            .groupby('user_id', as_index=False)['movie_id']
            .apply(lambda row: row.tolist())
        )
        df_user_top_n_last_movie_test.columns = ['user_id', 'recommended']

        return df_user_top_n_last_movie_test

    @classmethod
    def build_dataset(cls):
        df_movies, df_ratings, df_links = load_datasets()
        df_movie_features = cls.build_df_movie_features(df_movies)
        df_movie_features_preprocessed = cls.preprocess_df_movie_features(df_movie_features)

        df_user_last_movie_test = cls.build_df_user_last_movie_train(df_ratings)

        df_user_top_n_last_movie_test = cls.build_df_user_top_n_last_movie_test(df_ratings)

        split_timestamp = calculate_split_timestamp(df_ratings, CONFIG["test_size"])
        df_true_positives_test = build_df_true_positives(df_ratings, split_timestamp)

        return (
            df_movie_features_preprocessed,
            df_user_last_movie_test,
            df_user_top_n_last_movie_test,
            df_true_positives_test
        )



if __name__ == '__main__':
    (
        df_movie_features,
        df_user_last_movie_test,
        df_user_top_n_last_movie_test,
        df_true_positives_test
    ) =  KNNItemBasedBuildDataset.build_dataset()
    df_movie_features.to_pickle(CONFIG["df_movie_features_path"])
    df_user_last_movie_test.to_pickle(CONFIG["df_user_last_movie_test_path"])
    df_user_top_n_last_movie_test.to_pickle(CONFIG["df_user_top_n_last_movie_test_path"])
    df_true_positives_test.to_pickle(CONFIG["df_true_positives_test_path"])
