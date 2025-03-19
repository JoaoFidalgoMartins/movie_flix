import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# example http://127.0.0.1:8000/recommend?user_id=15&algo=popularity_ranker&top_n=3
# example http://127.0.0.1:8000/recommend?user_id=15&algo=knn_item_based_ranker&top_n=3
# example http://127.0.0.1:8000/recommend?user_id=15&algo=cat_boost_classifier_ranker&top_n=3