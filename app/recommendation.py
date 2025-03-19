from typing import List
import pandas as pd
from app.config import CONFIG

popularity_ranker_recs = pd.read_pickle(CONFIG["popularity_ranker_path"])
knn_item_based_ranker_recs = pd.read_pickle(CONFIG["knn_item_based_ranker_path"])
cat_boost_classifier_ranker_recs = pd.read_pickle(CONFIG["cat_boost_classifier_ranker_path"])

recs_batch = {
    'popularity_ranker': popularity_ranker_recs,
    'knn_item_based_ranker': knn_item_based_ranker_recs,
    'cat_boost_classifier_ranker': cat_boost_classifier_ranker_recs
}


def get_recommendations(user_id: int, algo: str, top_n: int) -> List[int]:
    return recs_batch[algo][recs_batch[algo].user_id == user_id]['recommended'].iloc[0][:top_n]


