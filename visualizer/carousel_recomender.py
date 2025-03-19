import pandas as pd
import random
from visualizer.visualizer import Visualizer, RecommendationCarousel
from utils.load_datasets import load_datasets
from visualizer.config import CONFIG


class CarouselRecommender:
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.visualizer = Visualizer()
        self.movies = None
        self.last_movie_seen = None
        self.last_top_n_movies_seen = None
        self.popularity_ranker_recs = None
        self.knn_item_based_ranker_recs = None
        self.cat_boost_classifier_ranker_recs = None


    def load(self):
        df_movies, df_ratings, df_links = load_datasets()
        self.movies = df_movies
        self.visualizer.load(df_movies, df_links, df_ratings)
        self.last_movie_seen = pd.read_pickle(CONFIG["last_movie_seen_path"])
        self.last_top_n_movies_seen = pd.read_pickle(CONFIG["last_top_n_movies_seen_path"])
        self.popularity_ranker_recs = pd.read_pickle(CONFIG["popularity_ranker_path"])
        self.knn_item_based_ranker_recs = pd.read_pickle(CONFIG["knn_item_based_ranker_path"])
        self.cat_boost_classifier_ranker_recs = pd.read_pickle(CONFIG["cat_boost_classifier_ranker_path"])

    @staticmethod
    def build_rec_carousel(title, recommendations, user_id):
        return RecommendationCarousel(
            title=title,
            recommended_items=recommendations[recommendations.user_id == user_id]['recommended'].iloc[0]
        )

    def recommend_user(self, user_id):
        id_last_movie_seen = self.last_movie_seen[self.last_movie_seen['user_id'] == user_id]['movie_id'].iloc[0]
        title_last_movie_seen = self.movies.loc[self.movies['movie_id'] == id_last_movie_seen]['title'].iloc[0]

        visualise_user = self.build_rec_carousel(
            title=f"Last 5 movies seen by user_id: {user_id}",
            recommendations=self.last_top_n_movies_seen,
            user_id=user_id
        )

        popularity_recs = self.build_rec_carousel(
            title=f"Movies you have to see at least once in your life!",
            recommendations=self.popularity_ranker_recs,
            user_id=user_id
        )

        last_movie_seen_recs = self.build_rec_carousel(
            title=f"Movies similar to: {title_last_movie_seen}",
            recommendations=self.knn_item_based_ranker_recs,
            user_id=user_id
        )

        best_for_you = self.build_rec_carousel(
            title=f"Best Movie for you!",
            recommendations=self.cat_boost_classifier_ranker_recs,
            user_id=user_id
        )


        list_recommendation_carousels = [
            visualise_user,
            last_movie_seen_recs,
            best_for_you,
            popularity_recs,
        ]

        self.visualizer.visualize_recommendation_carousels(list_recommendation_carousels)

    def visualize_random_user(self):
        available_user_test = list(
            set(self.last_top_n_movies_seen['user_id'].to_list())
            & set(self.cat_boost_classifier_ranker_recs['user_id'].to_list())
        )
        user_id = random.choice(available_user_test)
        self.recommend_user(user_id)



