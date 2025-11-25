# from fastapi import APIRouter, HTTPException, Body
# from pydantic import BaseModel
# from datetime import datetime
# from typing import List
# from services.database import price_collection
# from routes.price_prediction import predict_price

# router = APIRouter()

# # Schema for the input data
# class PriceEntry(BaseModel):
#     location: str
#     commodity: str
#     date: str  # Format YYYY-MM-DD
#     price: float

# @router.post("/api/add-price")
# async def add_price_entry(entry: PriceEntry):
#     try:
#         # 1. Save to MongoDB
#         entry_dict = entry.dict()
#         entry_dict["created_at"] = datetime.now()
#         price_collection.insert_one(entry_dict)

#         # 2. Get Model Prediction for comparison
#         # We call the existing prediction logic to get what the price *should* be
#         prediction_result = await predict_price(entry.commodity)
        
#         # We take the average of the 7-day forecast as the "Standard Model Price"
#         # This is robust even if dates don't match perfectly
#         forecast_prices = [item['price'] for item in prediction_result['forecast']]
#         model_price = sum(forecast_prices) / len(forecast_prices) if forecast_prices else entry.price

#         # 3. Logic: Check for Major Difference (e.g., > 15% deviation)
#         threshold = 1.15  # 15% buffer
#         is_inflation = False
#         message = "Price is within normal range."
#         action = "No action needed."

#         if entry.price > (model_price * threshold):
#             is_inflation = True
#             message = f"Warning: High Price Detected! (₹{entry.price} vs Model ₹{model_price:.2f})"
#             action = "RECOMMENDATION: RELEASE BUFFER STOCK IMMEDIATELY."

#         return {
#             "status": "success",
#             "saved_entry": entry,
#             "model_price": round(model_price, 2),
#             "is_inflation": is_inflation,
#             "message": message,
#             "action": action
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/api/admin/history/{location}/{commodity}")
# def get_admin_history(location: str, commodity: str):
#     # Fetch last 7 entries for this specific location/commodity from MongoDB
#     cursor = price_collection.find(
#         {"location": location, "commodity": commodity},
#         {"_id": 0, "date": 1, "price": 1}
#     ).sort("date", 1).limit(7)
    
#     data = list(cursor)
#     return data


from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from datetime import datetime
from typing import List
from services.database import price_collection
from routes.price_prediction import predict_price

router = APIRouter()

# UPDATED Schema: Added centre_id
class PriceEntry(BaseModel):
    location: str       # Stores centre_name (e.g., "Alipurduar")
    centre_id: str      # Stores centre_id (e.g., "472")
    commodity: str
    date: str
    price: float

@router.post("/api/add-price")
async def add_price_entry(entry: PriceEntry):
    if price_collection is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    try:
        # 1. Save to MongoDB
        entry_dict = entry.dict()
        entry_dict["created_at"] = datetime.now()
        price_collection.insert_one(entry_dict)

        # 2. Get Model Prediction for comparison
        prediction_result = await predict_price(entry.commodity)
        
        forecast_prices = [item['price'] for item in prediction_result['forecast']]
        model_price = sum(forecast_prices) / len(forecast_prices) if forecast_prices else entry.price

        # 3. Inflation Logic (15% threshold)
        threshold = 1.15
        is_inflation = False
        message = "Price is within normal range."
        action = "No action needed."

        if entry.price > (model_price * threshold):
            is_inflation = True
            message = f"Warning: High Price Detected! (₹{entry.price} vs Model ₹{model_price:.2f})"
            action = "RECOMMENDATION: RELEASE BUFFER STOCK IMMEDIATELY."

        return {
            "status": "success",
            "saved_entry": entry,
            "model_price": round(model_price, 2),
            "is_inflation": is_inflation,
            "message": message,
            "action": action
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/admin/history/{location}/{commodity}")
def get_admin_history(location: str, commodity: str):
    if price_collection is None:
        return []
        
    # Fetch history using the location name (standardized from the dropdown)
    cursor = price_collection.find(
        {"location": location, "commodity": commodity},
        {"_id": 0, "date": 1, "price": 1}
    ).sort("date", 1).limit(7)
    
    data = list(cursor)
    return data