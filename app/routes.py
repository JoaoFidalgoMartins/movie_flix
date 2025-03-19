from fastapi import APIRouter, Query
from .recommendation import get_recommendations

router = APIRouter()

@router.get("/recommend")
async def recommend(user_id: int = Query(..., description="The ID of the user"),
                    algo: str = Query(..., description="The algorithm to use for recommendations"),
                    top_n: int = Query(5, description="The number of recommendations to return")):
    recommendations = get_recommendations(user_id, algo, top_n)
    return {"recommendations": recommendations}
