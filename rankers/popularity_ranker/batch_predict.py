from build_dataset import build_dataset
from config import CONFIG

if __name__ == '__main__':
    df_ordered_popular_movie, df_true_positives_test, recommendations = build_dataset()
    df_ordered_popular_movie.to_pickle(CONFIG["df_ordered_popular_movie_path"])
    df_true_positives_test.to_pickle(CONFIG["df_true_positives_test_path"])
    recommendations.to_pickle(CONFIG["df_recommendations_test_path"])
