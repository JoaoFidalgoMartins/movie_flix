from utils.load_datasets import load_datasets
from rankers.cat_boost_classifier_ranker.build_dataset import CatBoostClassifierBuildDataset
from utils.data_utils import (
    get_unique_movie_genre,
    preprocess_df_movie,
    calculate_split_timestamp,
    split_train_test,
    build_df_true_positives
)
import pandas as pd
from rankers.cat_boost_classifier_ranker.config import CONFIG


if __name__ == '__main__':
    df_movies, df_ratings, df_links = load_datasets()

    unique_movie_genre = get_unique_movie_genre(df_movies)
    df_movies_preprocessed = preprocess_df_movie(df_movies, unique_movie_genre)
    df_base = pd.merge(df_ratings, df_movies_preprocessed, on='movie_id')
    df_base_sorted_user_id, df_base_sorted_movie_id = CatBoostClassifierBuildDataset.build_df_base_sorted(df_base)

    df_user_features = CatBoostClassifierBuildDataset.build_df_user_features(df_base_sorted_user_id, unique_movie_genre)
    df_movie_features = CatBoostClassifierBuildDataset.build_df_movie_features(df_base_sorted_movie_id)
    df_movie_features_static = CatBoostClassifierBuildDataset.build_df_movie_features_static(df_movies_preprocessed)
    idx_user_movie = CatBoostClassifierBuildDataset.build_idx_user_movie(df_ratings)

    df_user_movie_features = CatBoostClassifierBuildDataset.build_df_user_movie_features(
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

    df_cross_all_user_movie_features = CatBoostClassifierBuildDataset.build_df_cross_all_user_movie_features(
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
    idx_user_movie_positive_examples = CatBoostClassifierBuildDataset.build_idx_user_movie_positive_examples(df_ratings)
    idx_user_movie_negative_examples = CatBoostClassifierBuildDataset.build_idx_user_movie_negative_examples(df_ratings)

    df_movie_features.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/df_movie_features.pkl")
    df_movie_features_static.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/df_movie_features_static.pkl")
    df_user_features.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/df_user_features.pkl")
    idx_user_movie_positive_examples.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/idx_user_movie_positive_examples.pkl")
    idx_user_movie_negative_examples[['user_id', 'movie_id', 'timestamp', 'has_seen_movie']].to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/idx_user_movie_negative_examples.pkl")
    df_user_movie_features.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/df_user_movie_features.pkl")
    df_user_movie_features_train.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/df_user_movie_features_train.pkl")
    df_user_movie_features_test.to_pickle("notebooks/cat_boost_classifier_ranker/exploration_data/df_user_movie_features_test.pkl")