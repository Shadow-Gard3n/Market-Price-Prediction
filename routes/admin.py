import os
import numpy as np
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from services.database import price_collection
from tensorflow.keras.models import load_model

router = APIRouter()

# --- CONFIGURATION ---
MODELS_ROOT = 'Center_Price_Models'
PREDICTION_DAYS = 30  # Model needs exactly 30 days of context

# --- CONTROL PARAMETERS ---
Z_CRIT = 2.5
K_P = 100.0
K_D = 50.0

# Map Dropdown Values to Folder Names
COMMODITY_MAP = {
    "Groundnut_oil": "Groundnut", "Mustard_oil": "Mustard", "Vanaspati_oil": "Vanaspati",
    "Soya_oil": "Soya", "Palm_oil": "Palm", "Sunflower_oil": "Sunflower",
    "Tea_loose": "Tea", "Toor": "Toor", "Moong": "Moong", "Urad": "Urad",
    "Masoor": "Masoor", "Gram": "Gram", "Salt": "Salt", "Gur": "Gur",
    "Sugar": "Sugar", "Milk": "Milk", "Atta": "Atta", "Rice": "Rice",
    "Wheat": "Wheat", "Onion": "Onion", "Potato": "Potato", "Tomato": "Tomato"
}

class PriceEntry(BaseModel):
    location: str
    centre_id: str
    commodity: str
    date: str
    price: float

# --- HELPER FUNCTIONS ---

def get_history_from_db(center, commodity, limit=7):
    """Fetches last N days from MongoDB for stats calculation"""
    if price_collection is None: return []
    cursor = price_collection.find(
        {"location": center, "commodity": commodity},
        {"_id": 0, "price": 1, "date": 1}
    ).sort("date", -1).limit(limit)
    return list(cursor)

def get_seed_from_db(commodity, center, target_date_str):
    """
    Fetches the latest 30 days of data up to and including the target date.
    """
    if price_collection is None:
        print("❌ DB Connection is None")
        return None

    # CORRECTED QUERY:
    # 1. Look for dates less than or equal ($lte) to the target date
    # 2. Sort Descending (Newest first) to get the immediate history
    # 3. Limit to 30
    cursor = price_collection.find(
        {
            "location": center, 
            "commodity": commodity,
            "date": {"$lte": target_date_str}  # Changed from $lt to $lte
        },
        {"_id": 0, "price": 1, "date": 1}
    ).sort("date", -1).limit(PREDICTION_DAYS)

    data = list(cursor)

    # Debug print to help you see what's happening in the console
    print(f"🔍 Fetching seed for {center}-{commodity}. Found {len(data)} rows.")

    if len(data) < PREDICTION_DAYS:
        print(f"⚠️ Insufficient data. Needed {PREDICTION_DAYS}, found {len(data)}")
        return None

    # Sort Ascending (Oldest -> Newest) for the LSTM input
    data.sort(key=lambda x: x['date']) # Python sort is safer than relying on DB reverse here
    
    # Extract prices
    prices = np.array([d['price'] for d in data])
    return prices

# --- MAIN ROUTE ---
@router.post("/api/add-price")
async def add_price_entry(entry: PriceEntry):
    if price_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        # 1. Save/Upsert Entry
        filter_query = {"location": entry.location, "commodity": entry.commodity, "date": entry.date}
        update_query = {
            "$set": {"centre_id": entry.centre_id, "price": entry.price, "created_at": datetime.now()}
        }
        price_collection.update_one(filter_query, update_query, upsert=True)

        # 2. Get History & Filter Outliers (THE FIX)
        history_docs = get_history_from_db(entry.location, entry.commodity, limit=14) # Look back 2 weeks
        raw_prices = [d['price'] for d in history_docs]
        
        # Add current price if not in history yet
        if not any(d['date'] == entry.date for d in history_docs):
            raw_prices.append(entry.price)

        # --- STEP 2.1: DATA CLEANING (Remove crazy outliers) ---
        # If we have [40, 42, 41, 400], the 400 ruins the math. We remove it.
        median_price = np.median(raw_prices)
        clean_prices = [p for p in raw_prices if p <= (median_price * 2) and p >= (median_price * 0.5)]
        
        if len(clean_prices) < 2:
            mu = entry.price
            sigma = 0.0
        else:
            mu = np.mean(clean_prices)
            sigma = np.std(clean_prices, ddof=1)

        # --- STEP 2.2: TIGHTER LOGIC ---
        # 1. Statistical Limit (Mean + 2 Sigma)
        stat_ucl = mu + (2.0 * sigma) 
        
        # 2. Hard Percentage Limit (Max 20% increase allowed from average)
        # This prevents the "442 UCL" issue. UCL can never exceed 20% of mean.
        hard_cap_ucl = mu * 1.20 
        
        # The final Upper Control Limit is the LOWER of the two.
        # This makes the system sensitive.
        ucl = min(stat_ucl, hard_cap_ucl) 
        
        # If sigma is tiny (e.g. price is stable at 40, 40, 40), give a small buffer
        if ucl < mu * 1.05: 
            ucl = mu * 1.05

        # 3. Forecast Logic
        forecast_prices = []
        forecast_dates = []
        is_alarm = False
        action_msg = "Market is Stable."
        alarm_msg = f"Prices are normal. (Limit: ₹{ucl:.2f})"
        
        # ... (Model Prediction Code stays the same) ...
        try:
            folder_name = COMMODITY_MAP.get(entry.commodity, entry.commodity.capitalize())
            model_path = os.path.join(MODELS_ROOT, folder_name, entry.location, 'model.keras')
            scaler_path = os.path.join(MODELS_ROOT, folder_name, entry.location, 'scaler.pkl')
            
            if os.path.exists(model_path):
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                seed_values = get_seed_from_db(entry.commodity, entry.location, entry.date)
                
                if seed_values is not None:
                    curr_batch = scaler.transform(seed_values.reshape(-1, 1)).reshape(1, PREDICTION_DAYS, 1)
                    for i in range(7):
                        pred = model.predict(curr_batch, verbose=0)[0]
                        val = scaler.inverse_transform([[pred[0]]])[0][0]
                        forecast_prices.append(float(val))
                        curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)
                        f_date = pd.to_datetime(entry.date) + pd.Timedelta(days=i+1)
                        forecast_dates.append(f_date.strftime("%Y-%m-%d"))
        except Exception:
            pass

        # 4. DECISION LOGIC (Corrected)
        
        # Check 1: Is Current Price too high?
        if entry.price > ucl:
            is_alarm = True
            gap = entry.price - mu
            # release = 100 * gap (Simple: 100 tonnes per Rupee over limit)
            q_release = max(10, 100.0 * gap) 
            alarm_msg = f"ALARM: Current Price ₹{entry.price} > Limit ₹{ucl:.2f}"
            action_msg = f"RECOMMENDATION: IMMEDIATE RELEASE OF {q_release:.0f} TONNES"

        # Check 2: Are Future Prices too high? (Pre-emptive)
        elif forecast_prices:
            for p_pred in forecast_prices:
                if p_pred > ucl:
                    is_alarm = True
                    gap = p_pred - mu
                    q_release = max(10, 80.0 * gap) # Release slightly less for future prediction
                    alarm_msg = f"WARNING: Forecast hits ₹{p_pred:.2f} (Limit: ₹{ucl:.2f})"
                    action_msg = f"RECOMMENDATION: PLAN RELEASE OF {q_release:.0f} TONNES"
                    break

        return {
            "status": "success",
            "is_inflation": is_alarm,
            "message": alarm_msg,
            "action": action_msg,
            "forecast": [{"date": d, "price": round(p, 2)} for d, p in zip(forecast_dates, forecast_prices)]
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/admin/history/{location}/{commodity}")
def get_admin_history(location: str, commodity: str):
    if price_collection is None: return []
    # Fetch 30 days history so the chart looks nice
    cursor = price_collection.find(
        {"location": location, "commodity": commodity},
        {"_id": 0, "date": 1, "price": 1}
    ).sort("date", -1).limit(30) 
    return list(cursor)

# from fastapi import APIRouter, HTTPException, Body
# from pydantic import BaseModel
# from datetime import datetime
# from typing import List
# from services.database import price_collection
# from routes.price_prediction import predict_price

# router = APIRouter()

# # UPDATED Schema: Added centre_id
# class PriceEntry(BaseModel):
#     location: str       # Stores centre_name (e.g., "Alipurduar")
#     centre_id: str      # Stores centre_id (e.g., "472")
#     commodity: str
#     date: str
#     price: float

# @router.post("/api/add-price")
# async def add_price_entry(entry: PriceEntry):
#     if price_collection is None:
#         raise HTTPException(status_code=503, detail="Database connection unavailable")

#     try:
#         # 1. Save to MongoDB
#         entry_dict = entry.dict()
#         entry_dict["created_at"] = datetime.now()
#         price_collection.insert_one(entry_dict)

#         # 2. Get Model Prediction for comparison
#         prediction_result = await predict_price(entry.commodity)
        
#         forecast_prices = [item['price'] for item in prediction_result['forecast']]
#         model_price = sum(forecast_prices) / len(forecast_prices) if forecast_prices else entry.price

#         # 3. Inflation Logic (15% threshold)
#         threshold = 1.15
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
#     if price_collection is None:
#         return []
        
#     # Fetch history using the location name (standardized from the dropdown)
#     cursor = price_collection.find(
#         {"location": location, "commodity": commodity},
#         {"_id": 0, "date": 1, "price": 1}
#     ).sort("date", 1).limit(7)
    
#     data = list(cursor)
#     return data