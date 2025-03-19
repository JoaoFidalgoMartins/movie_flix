from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


class KNNItemBasedRanker:
    def __init__(self):
        self.algo = NearestNeighbors(n_neighbors=50, metric='cosine')

    def fit(self, df_train):
        self.algo.fit(df_train)

    def save(self):
        pass

    def load(self):
        pass

    def recommend_movie_batch(self, df_movie_features, top_n=5):
        recommendations_idx = self.algo.kneighbors(df_movie_features, n_neighbors=top_n+1)[1]
        # prevent recommending itself
        recommendations_idx = np.array([
            [x for x in row if x != i][:top_n]
            for i, row in enumerate(recommendations_idx)
        ])

        # map index to item_id
        map_item_ids = {idx: value for idx, value in enumerate(df_movie_features.index)}
        recommended = np.vectorize(map_item_ids.get)(recommendations_idx)

        df_recommendations_movie = df_movie_features.reset_index()[['movie_id']]
        df_recommendations_movie.loc[:, 'recommended'] = pd.DataFrame(recommended).apply(lambda row: row.tolist(), axis=1)

        return df_recommendations_movie

    def recommend_user_batch(self, df_recommendations_movie, df_user_last_movie):
        return df_user_last_movie.merge(df_recommendations_movie, on='movie_id').drop(columns=["movie_id"])

