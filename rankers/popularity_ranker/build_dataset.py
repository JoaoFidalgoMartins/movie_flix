from utils.load_datasets import load_datasets
from utils.data_utils import calculate_split_timestamp, split_train_test, build_df_true_positives
from rankers.popularity_ranker.config import CONFIG
import pandas as pd


def build_dataset():
    df_movies, df_ratings, df_links = load_datasets()
    split_timestamp = calculate_split_timestamp(df_ratings, CONFIG["test_size"])
    df_ratings_train, df_ratings_test = split_train_test(df_ratings, split_timestamp)

    df_popular = df_ratings_train.groupby('movie_id', as_index=False)[['user_id']].nunique()
    df_popular.columns = ['movie_id', 'count_distinct_user_id']
    df_popular = df_popular.sort_values('count_distinct_user_id', ascending=False)
    df_ordered_popular_movie = df_popular[['movie_id']].reset_index(drop=True)

    df_true_positives_test = build_df_true_positives(df_ratings, split_timestamp)

    recs_top_n = df_ordered_popular_movie[:5]['movie_id'].to_list()
    recommended = [recs_top_n for i in range(len(df_true_positives_test))]
    recommendations = df_true_positives_test[['user_id']]
    recommendations.loc[:, 'recommended'] = pd.DataFrame(recommended).apply(lambda row: row.tolist(), axis=1)

    return df_ordered_popular_movie, df_true_positives_test, recommendations


if __name__ == '__main__':
    df_ordered_popular_movie, df_true_positives_test, recommendations = build_dataset()
    df_ordered_popular_movie.to_pickle(CONFIG["df_ordered_popular_movie_path"])
    df_true_positives_test.to_pickle(CONFIG["df_true_positives_test_path"])
    recommendations.to_pickle(CONFIG["df_recommendations_test_path"])
