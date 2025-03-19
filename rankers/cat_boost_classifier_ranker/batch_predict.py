from rankers.cat_boost_classifier_ranker.ranker import CatBoostClassifierRanker
from rankers.cat_boost_classifier_ranker.build_dataset import CatBoostClassifierBuildDataset
from rankers.cat_boost_classifier_ranker.config import CONFIG

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

    algo = CatBoostClassifierRanker()
    algo.fit(df_user_movie_features_train, df_user_movie_features_test)

    recommendations = algo.recommend_batch(df_cross_all_user_movie_features_test)
    recommendations.to_pickle(CONFIG["df_recommendations_test_path"])
