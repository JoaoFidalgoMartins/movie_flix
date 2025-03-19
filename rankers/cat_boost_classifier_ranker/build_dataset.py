import pandas as pd
import random

from utils.load_datasets import load_datasets
from utils.data_utils import (
    get_unique_movie_genre,
    preprocess_df_movie,
    calculate_split_timestamp,
    split_train_test,
    build_df_true_positives,
)
from rankers.cat_boost_classifier_ranker.config import CONFIG
from sklearn.preprocessing import StandardScaler


class CatBoostClassifierBuildDataset:
    def __init__(self):
        pass

    @staticmethod
    def build_df_base_sorted(df_base):
        # for cumsum the dfs need to be sorted
        df_base_sorted_user_id = df_base.sort_values(['user_id', 'timestamp', 'movie_id']).reset_index(drop=True)
        df_base_sorted_movie_id = df_base.sort_values(['movie_id', 'timestamp', 'user_id']).reset_index(drop=True)
        return df_base_sorted_user_id, df_base_sorted_movie_id

    @staticmethod
    def build_df_user_total_movie_count(df_base_sorted_user_id):
        df_user_total_movie_count = (
            df_base_sorted_user_id
            .groupby(['user_id', 'timestamp'], as_index=False)['movie_id']
            .count()
            .rename(columns={'movie_id': 'user_total_movie_count'})
        )
        df_user_total_movie_count = (
            df_user_total_movie_count[['user_id', 'user_total_movie_count']]
            .groupby(['user_id'])
            .cumsum()
        )
        return df_user_total_movie_count

    @staticmethod
    def build_df_user_movie_percent_by_genre(df_base_sorted_user_id, unique_movie_genre, df_user_total_movie_count):
        df_user_movie_percent_by_genre = (
            df_base_sorted_user_id
            .groupby(['user_id', 'timestamp'], as_index=False)[unique_movie_genre]
            .sum()
        )
        df_user_movie_percent_by_genre = (
            df_user_movie_percent_by_genre[['user_id'] + unique_movie_genre]
            .groupby(['user_id'])
            .cumsum()
        )
        df_user_movie_percent_by_genre = (
            df_user_movie_percent_by_genre
            .divide(df_user_total_movie_count['user_total_movie_count'], axis=0)
        )

        df_user_movie_percent_by_genre.columns = [f'user_movie_percent_{genre}' for genre in unique_movie_genre]

        return df_user_movie_percent_by_genre

    @staticmethod
    def build_df_user_avg_rating(df_base_sorted_user_id):
        df_sum_rating = (
            df_base_sorted_user_id
            .groupby(['user_id', 'timestamp'], as_index=False)['rating']
            .sum()
        )
    
        df_sum_rating = (
            df_sum_rating[['user_id', 'rating']]
            .groupby(['user_id'])
            .cumsum()
        )
    
        df_count_rating = (
            df_base_sorted_user_id
            .groupby(['user_id', 'timestamp'], as_index=False)['rating']
            .count()
        )
        df_count_rating = (
            df_count_rating[['user_id', 'rating']]
            .groupby(['user_id'])
            .cumsum()
        )
    
        df_user_avg_rating = df_sum_rating / df_count_rating
        df_user_avg_rating.columns = ['user_avg_rating']
    
        return df_user_avg_rating

    @staticmethod
    def build_df_user_avg_year(df_base_sorted_user_id):
        df_sum_year = (
            df_base_sorted_user_id
            .groupby(['user_id', 'timestamp'], as_index=False)['year']
            .sum()
        )
        df_sum_year = (
            df_sum_year[['user_id', 'year']]
            .groupby(['user_id'])
            .cumsum()
        )
    
        df_count_year = (
            df_base_sorted_user_id
            .groupby(['user_id', 'timestamp'], as_index=False)['year']
            .count()
        )
        df_count_year = (
            df_count_year[['user_id', 'year']]
            .groupby(['user_id'])
            .cumsum()
        )
    
        df_user_avg_year = df_sum_year / df_count_year
        df_user_avg_year.columns = ['user_avg_year']
    
        return df_user_avg_year

    @classmethod
    def build_df_user_features(cls, df_base_sorted_user_id, unique_movie_genre):
        idx_user = df_base_sorted_user_id[['user_id', 'timestamp']].drop_duplicates().reset_index(drop=True)
    
        df_user_avg_rating = cls.build_df_user_avg_rating(df_base_sorted_user_id)
        df_user_avg_year = cls.build_df_user_avg_year(df_base_sorted_user_id)
        df_user_total_movie_count = cls.build_df_user_total_movie_count(df_base_sorted_user_id)
        df_user_movie_percent_by_genre = cls.build_df_user_movie_percent_by_genre(
            df_base_sorted_user_id, 
            unique_movie_genre,
            df_user_total_movie_count
        )
    
        df_user = pd.concat([
            idx_user,
            df_user_avg_rating,
            df_user_avg_year,
            df_user_total_movie_count,
            df_user_movie_percent_by_genre,
        ], axis=1)
        return df_user

    @staticmethod
    def build_df_movie_total_user_count(df_base_sorted_movie_id):
        df_movie_total_user_count = (
            df_base_sorted_movie_id
            .groupby(['movie_id', 'timestamp'], as_index=False)['user_id']
            .count()
            .rename(columns={'user_id': 'movie_total_user_count'})
        )
        df_movie_total_user_count = (
            df_movie_total_user_count[['movie_id', 'movie_total_user_count']]
            .groupby(['movie_id'])
            .cumsum()
        )
        return df_movie_total_user_count

    @classmethod
    def build_df_movie_features(cls, df_base_sorted_movie_id):
        idx_movie = df_base_sorted_movie_id[['movie_id', 'timestamp']].drop_duplicates().reset_index(drop=True)
    
        df_movie_total_user_count = cls.build_df_movie_total_user_count(df_base_sorted_movie_id)
        df_movie_avg_rating = cls.build_df_movie_avg_rating(df_base_sorted_movie_id)
    
        df_movie_features = (
            pd.concat([
                idx_movie,
                df_movie_total_user_count,
                df_movie_avg_rating,
            ], axis=1)
        )
    
        return df_movie_features

    @staticmethod
    def build_df_movie_features_static(df_movies_preprocessed):
        movie_features_static = df_movies_preprocessed
        df_movies_preprocessed.columns = ['movie_id'] + [f"movie_{col}" for col in movie_features_static.drop('movie_id', axis=1).columns]
        return movie_features_static

    @staticmethod
    def build_df_movie_avg_rating(df_base_sorted_movie_id):
        df_movie_sum_rating = df_base_sorted_movie_id.groupby(['movie_id', 'timestamp'], as_index=False)['rating'].sum()
        df_movie_sum_rating = df_movie_sum_rating[['movie_id', 'rating']].groupby(['movie_id']).cumsum()
        df_movie_count_rating = df_base_sorted_movie_id.groupby(['movie_id', 'timestamp'], as_index=False)['rating'].count()
        df_movie_count_rating = df_movie_count_rating[['movie_id', 'rating']].groupby(['movie_id']).cumsum()
        df_movie_avg_rating = df_movie_sum_rating / df_movie_count_rating
        df_movie_avg_rating.columns = ['movie_avg_rating']
        return df_movie_avg_rating

    @staticmethod
    def build_idx_user_movie_positive_examples(df_ratings):
        idx_user_movie_positive_examples = (
            df_ratings
            .sort_values(['user_id', 'timestamp', 'movie_id']).reset_index(drop=True)
            [['user_id', 'movie_id', 'timestamp']]
            .reset_index(drop=True)
        )
        idx_user_movie_positive_examples['has_seen_movie'] = 1
        return idx_user_movie_positive_examples

    @classmethod
    def build_idx_user_movie_negative_examples(cls, df_ratings):
        movie_unique = df_ratings['movie_id'].unique()
        movies_seen_by_user = build_df_true_positives(df_ratings, split_timestamp=0)
    
        idx_user_movie_negative_example = df_ratings[['user_id', 'timestamp']]
        idx_user_movie_negative_example = idx_user_movie_negative_example.merge(movies_seen_by_user, on='user_id')
        idx_user_movie_negative_example['user_has_seen_random_movie'] = True
        while idx_user_movie_negative_example.user_has_seen_random_movie.max():
            idx_user_movie_negative_example.loc[idx_user_movie_negative_example.user_has_seen_random_movie, 'movie_id'] = \
                random.choices(
                    movie_unique,
                    k=len(idx_user_movie_negative_example.loc[idx_user_movie_negative_example.user_has_seen_random_movie])
                )
    
            idx_user_movie_negative_example['user_has_seen_random_movie'] = \
                idx_user_movie_negative_example.apply(lambda x: x['movie_id'] in x['true_positives'], axis=1)
        idx_user_movie_negative_example = (
            idx_user_movie_negative_example
            .drop(columns=['true_positives', 'user_has_seen_random_movie'])
        )
        idx_user_movie_negative_example['movie_id'] = idx_user_movie_negative_example['movie_id'].astype('int')
        idx_user_movie_negative_example['has_seen_movie'] = 0
        return idx_user_movie_negative_example

    @classmethod
    def build_idx_user_movie(cls, df_ratings):
        idx_user_movie_positive_examples = cls.build_idx_user_movie_positive_examples(df_ratings)
        idx_user_movie_negative_examples = cls.build_idx_user_movie_negative_examples(df_ratings)
        idx_user_movie = pd.concat([
            idx_user_movie_positive_examples,
            idx_user_movie_negative_examples
        ])
        return idx_user_movie

    @staticmethod
    def build_df_user_movie_features(idx_user_movie, df_user_features, df_movie_features, df_movie_features_static):
        df_join_user = (
            pd.merge_asof(
                idx_user_movie.drop('has_seen_movie', axis=1).sort_values(by='timestamp'),
                df_user_features.sort_values(by='timestamp'),
                on='timestamp',
                by='user_id',
                direction='backward',
                allow_exact_matches=False
            )
        )
    
        df_join_movie = (
            pd.merge_asof(
                idx_user_movie.drop('has_seen_movie', axis=1).sort_values(by='timestamp'),
                df_movie_features.sort_values(by='timestamp'),
                on='timestamp',
                by='movie_id',
                direction='backward',
                allow_exact_matches=False
            )
            .merge(df_movie_features_static, on='movie_id', how='left')
        )
    
        df_user_movie_features = (
            idx_user_movie
            .merge(df_join_user, on=['user_id', 'movie_id', 'timestamp'], how='left')
            .merge(df_join_movie, on=['user_id', 'movie_id', 'timestamp'], how='left')
        )
        return df_user_movie_features

    @staticmethod
    def build_df_cross_all_user_movie_features(
            df_user_features, df_movie_features, df_movie_features_static, df_user_movie_features
    ):
        # get last user state by timestamp
        dataset_user_last_timestamp = (
            df_user_features
            .groupby('user_id', as_index=False)[['timestamp']].max()
            .merge(df_user_features, on=['user_id', 'timestamp'], how='inner')
            .reset_index(drop=True)
        )
    
        dataset_movie_last_timestamp = (
            df_movie_features
            .groupby('movie_id', as_index=False)[['timestamp']].max()
            .merge(df_movie_features, on=['movie_id', 'timestamp'], how='inner')
            .merge(df_movie_features_static, on='movie_id', how='left')
            .drop('timestamp', axis=1)
            .reset_index(drop=True)
        )
    
        # cross join all users vs all items
        dataset_user_last_timestamp['key'] = 0
        dataset_movie_last_timestamp['key'] = 0
    
        df_cross_all_user_movie_features = (
            dataset_user_last_timestamp
            .merge(dataset_movie_last_timestamp, on='key',how='outer')
            .drop('key', axis=1)
        )

        cols = df_user_movie_features.drop('has_seen_movie', axis=1).columns
        df_cross_all_user_movie_features = df_cross_all_user_movie_features[cols]
        return df_cross_all_user_movie_features

    @classmethod
    def build_datasets(cls):
        df_movies, df_ratings, df_links = load_datasets()

        unique_movie_genre = get_unique_movie_genre(df_movies)
        df_movies_preprocessed = preprocess_df_movie(df_movies, unique_movie_genre)
        df_base = pd.merge(df_ratings, df_movies_preprocessed, on='movie_id')
        df_base_sorted_user_id, df_base_sorted_movie_id = cls.build_df_base_sorted(df_base)
    
        df_user_features = cls.build_df_user_features(df_base_sorted_user_id, unique_movie_genre)
        df_movie_features = cls.build_df_movie_features(df_base_sorted_movie_id)
        df_movie_features_static = cls.build_df_movie_features_static(df_movies_preprocessed)
        idx_user_movie = cls.build_idx_user_movie(df_ratings)
    
        df_user_movie_features = cls.build_df_user_movie_features(
            idx_user_movie, 
            df_user_features, 
            df_movie_features, 
            df_movie_features_static
        )

        split_timestamp = calculate_split_timestamp(df_ratings, CONFIG["test_size"])
        df_user_movie_features_train, df_user_movie_features_test = split_train_test(
            df_user_movie_features,
            split_timestamp
        )
    
        df_cross_all_user_movie_features = cls.build_df_cross_all_user_movie_features(
            df_user_features, 
            df_movie_features, 
            df_movie_features_static,
            df_user_movie_features
        )

        df_cross_all_user_movie_features_train, df_cross_all_user_movie_features_test = split_train_test(
            df_cross_all_user_movie_features,
            split_timestamp
        )

        df_true_positives_test = build_df_true_positives(df_ratings, split_timestamp)
    
        return (
            df_user_movie_features_train,
            df_user_movie_features_test,
            df_cross_all_user_movie_features_test,
            df_true_positives_test
        )



if __name__ == '__main__':
    (
        df_user_movie_features_train,
        df_user_movie_features_test,
        df_cross_all_user_movie_features_test,
        df_true_positives_test
    ) = CatBoostClassifierBuildDataset.build_datasets()

    df_user_movie_features_train.to_pickle(CONFIG["df_user_movie_features_train_path"])
    df_user_movie_features_test.to_pickle(CONFIG["df_user_movie_features_test_path"])
    df_cross_all_user_movie_features_test.to_pickle(CONFIG["df_cross_all_user_movie_features_test_path"])
    df_true_positives_test.to_pickle(CONFIG["df_true_positives_test_path"])
