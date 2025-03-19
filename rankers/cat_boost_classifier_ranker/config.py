CONFIG = {
    "directory_path": "/Users/j.fidalgo/PycharmProjects/movie_flix",
    "df_recommendations_test_path": "output/cat_boost_classifier_ranker/df_recommendations_test.pkl",
    "df_user_movie_features_train_path": "output/cat_boost_classifier_ranker/df_user_movie_features_train.pkl",
    "df_user_movie_features_test_path": "output/cat_boost_classifier_ranker/df_user_movie_features_test.pkl",
    "df_cross_all_user_movie_features_test_path" : "output/cat_boost_classifier_ranker/df_cross_all_user_movie_features_test.pkl",
    "df_true_positives_test_path": "output/cat_boost_classifier_ranker/df_true_positives_test.pkl",
    "model_path": "output/cat_boost_classifier_ranker/cat_boost_classifier_ranker.cbm",
    "params": {
        'iterations': 425,
         'learning_rate': 0.06023131826303992,
         'depth': 10,
         'l2_leaf_reg': 4.55,
         'subsample': 0.85,
         'loss_function': 'Logloss',
         'eval_metric': 'AUC'
    },
    "test_size": 0.2,
    "target": 'has_seen_movie',
    "cat_features": ['movie_id', 'user_id'],
}

