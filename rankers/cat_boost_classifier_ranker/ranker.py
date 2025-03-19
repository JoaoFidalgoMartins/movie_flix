from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd

from rankers.cat_boost_classifier_ranker.config import CONFIG
from metrics.metrics import Metrics


class CatBoostClassifierRanker:
    def __init__(self):
        self.algo = CatBoostClassifier(**CONFIG['params'])

    def save(self):
        self.algo.save_model(CONFIG['model_path'])

    def load(self):
        self.algo.load_model(CONFIG['model_path'])

    def fit(self, df_train, df_test):
        target = CONFIG['target']
        cat_features = CONFIG['cat_features']

        features = df_train.drop(target, axis=1).columns.to_list()

        # make sure df are sorted by timestamp if TimeSeriesSplit is used
        df_train = df_train.sort_values(['timestamp', 'user_id', 'movie_id']).reset_index(drop=True)
        df_test = df_test.sort_values(['timestamp', 'user_id', 'movie_id']).reset_index(drop=True)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = df_train[features], df_test[features], df_train[target], df_test[target]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        self.algo.fit(train_pool, eval_set=test_pool, plot=False, early_stopping_rounds=50)

    @staticmethod
    def eval(recommendations, true_positives, k):
        return Metrics.evaluate(recommendations, true_positives, k)

    def recommend_batch(self, df_all_user_movie_features, top_n=5):
        predictions = self.algo.predict_proba(df_all_user_movie_features)[:, 1]

        df_recs = df_all_user_movie_features[['user_id', 'movie_id']]
        df_recs.loc[:, 'prediction'] = predictions
        df_recs = df_recs.pivot(index='user_id', columns='movie_id', values='prediction')

        recommendations_idx = np.argsort(df_recs.values, axis=1)[:, ::-1][:, :top_n]

        # map index to item_id
        map_item_ids = {idx: value for idx, value in enumerate(df_recs.columns)}
        recommendations = np.vectorize(map_item_ids.get)(recommendations_idx)

        # Convert back to DataFrame for readability
        recommendations = pd.DataFrame(recommendations, index=df_recs.index).apply(list, axis=1).reset_index()
        recommendations.columns = ['user_id', 'recommended']

        return recommendations


