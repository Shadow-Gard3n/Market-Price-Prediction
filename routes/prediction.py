from fastapi import APIRouter, HTTPException, Query
from typing import List
from services.market_services import get_market_prediction
from services.marketTracking_services import fetch_market_data
from schemas.marketTracker_schemas import MarketPriceRequest, MarketPriceData

router = APIRouter()

    
@router.get("/api/predict/{commodity}")
def predict_commodity_price(commodity: str):
    result = get_market_prediction(commodity)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.post(
    "/api/marketPrice",
    response_model=List[MarketPriceData],
    summary="Fetch Agricultural Market Prices",
    description="Retrieves daily market price data for a specific commodity, state, and APMC over the last 7 days."
)
async def get_market_price(request: MarketPriceRequest):
    market_data = await fetch_market_data(request)
    return market_data

