from rankers.knn_item_based_ranker.ranker import KNNItemBasedRanker
from rankers.knn_item_based_ranker.build_dataset import KNNItemBasedBuildDataset

from rankers.knn_item_based_ranker.config import CONFIG

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

    algo = KNNItemBasedRanker()
    algo.fit(df_movie_features)

    df_recommendations_movie = algo.recommend_movie_batch(df_movie_features)
    df_recommendations_movie.to_pickle(CONFIG["df_recommendations_movie_path"])

    df_recommendations_user = algo.recommend_user_batch(df_recommendations_movie, df_user_last_movie_test)
    df_recommendations_user.to_pickle(CONFIG["df_recommendations_user_test_path"])


