import os
import numpy as np
import joblib
import pandas as pd
import glob
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from services.database import price_collection
from tensorflow.keras.models import load_model

router = APIRouter()

# --- CONFIGURATION ---
MODELS_ROOT = 'models'
DATASET_ROOT = 'Center_Price_Dataset'
PREDICTION_DAYS = 30

# --- CONTROL PARAMETERS (From your script) ---
Z_CRIT = 2.5   # Sigma threshold
K_P = 100.0    # Proportional Gain
K_D = 50.0     # Derivative Gain

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

def get_history_from_db(center, commodity):
    """Fetches last 7 days from MongoDB for stats calculation"""
    if price_collection is None: return []
    cursor = price_collection.find(
        {"location": center, "commodity": commodity},
        {"_id": 0, "price": 1, "date": 1}
    ).sort("date", -1).limit(7)
    return list(cursor)

def get_seed_sequence(commodity, center, target_date_str):
    """
    Finds the 30 days of data BEFORE the target date to seed the prediction.
    This fixes the 'Date is in the past' error.
    """
    folder_name = COMMODITY_MAP.get(commodity, commodity.capitalize())
    search_pattern = f"{DATASET_ROOT}/*{folder_name.lower()}*.csv"
    files = glob.glob(search_pattern)
    
    if not files: return None, None
    
    try:
        df = pd.read_csv(files[0])
        df.columns = [c.lower().strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Filter for center
        df_center = df[df['centre_name'] == center]
        if len(df_center) < PREDICTION_DAYS: return None, None
        
        target_dt = pd.to_datetime(target_date_str)
        
        # Get data STRICTLY BEFORE target date
        past_data = df_center[df_center['date'] < target_dt]
        
        if len(past_data) < PREDICTION_DAYS:
            # Fallback: If targeting start of file, just take first 30
            if len(df_center) >= PREDICTION_DAYS:
                return df_center['date'].iloc[-1], df_center['price'].values[-PREDICTION_DAYS:]
            return None, None

        return past_data['date'].iloc[-1], past_data['price'].values[-PREDICTION_DAYS:]
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

# --- MAIN ROUTE ---
@router.post("/api/add-price")
async def add_price_entry(entry: PriceEntry):
    if price_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        # 1. Save/Upsert Entry (Update if same date/loc/comm exists)
        filter_query = {"location": entry.location, "commodity": entry.commodity, "date": entry.date}
        update_query = {
            "$set": {"centre_id": entry.centre_id, "price": entry.price, "created_at": datetime.now()}
        }
        price_collection.update_one(filter_query, update_query, upsert=True)

        # 2. Calculate Historical Stats (Mu, Sigma, UCL)
        history_docs = get_history_from_db(entry.location, entry.commodity)
        
        # Combine DB history + Current Entry for calculation
        price_values = [d['price'] for d in history_docs]
        price_values.append(entry.price) 
        
        if len(price_values) < 2:
            # Not enough data for Std Dev
            mu = entry.price
            sigma = 0.0
        else:
            arr = np.array(price_values)
            mu = np.mean(arr)
            sigma = np.std(arr, ddof=1)
        
        ucl = mu + (Z_CRIT * sigma)

        # 3. Generate 7-Day Forecast (Center-Specific)
        forecast_prices = []
        forecast_dates = []
        
        try:
            folder_name = COMMODITY_MAP.get(entry.commodity, entry.commodity.capitalize())
            model_path = os.path.join(MODELS_ROOT, folder_name, entry.location, 'model.keras')
            scaler_path = os.path.join(MODELS_ROOT, folder_name, entry.location, 'scaler.pkl')

            if os.path.exists(model_path):
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                
                # Get seed data based on Input Date
                last_date, seed_values = get_seed_sequence(entry.commodity, entry.location, entry.date)
                
                if seed_values is not None:
                    # Prepare Batch
                    curr_batch = scaler.transform(seed_values.reshape(-1, 1)).reshape(1, PREDICTION_DAYS, 1)
                    
                    # Predict 7 days into the future from Target Date
                    for i in range(7):
                        pred = model.predict(curr_batch, verbose=0)[0]
                        val = scaler.inverse_transform([[pred[0]]])[0][0]
                        forecast_prices.append(float(val))
                        
                        # Update batch
                        curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)
                        
                        # Calc date
                        f_date = pd.to_datetime(entry.date) + pd.Timedelta(days=i+1)
                        forecast_dates.append(f_date.strftime("%Y-%m-%d"))
                        
        except Exception as e:
            print(f"Prediction failed: {e}")

        # 4. Run Your PD Control Logic
        is_alarm = False
        q_release = 0.0
        alarm_msg = f"Market is stable. (UCL: ₹{ucl:.2f})"
        action_msg = "No Action Needed."
        
        # Logic: Check Forecast against UCL
        if forecast_prices:
            for k, p_pred in enumerate(forecast_prices):
                if p_pred > ucl:
                    is_alarm = True
                    
                    # Use previous day forecast OR current actual price as P(k-1)
                    p_prev = forecast_prices[k-1] if k > 0 else entry.price
                    
                    # PD Control Formula
                    delta_p = p_pred - mu       # Price Gap
                    rate_change = p_pred - p_prev  # Derivative
                    
                    q_release = (K_P * delta_p) + (K_D * rate_change)
                    q_release = max(0, q_release) # Ensure positive
                    
                    alarm_msg = f"ALARM: Predicted ₹{p_pred:.2f} on {forecast_dates[k]} > UCL ₹{ucl:.2f}"
                    action_msg = f"RECOMMENDATION: RELEASE {q_release:.2f} TONNES IMMEDIATELY."
                    break
        
        # Logic: Fallback if current price is already too high
        if not is_alarm and entry.price > ucl and sigma > 0:
             is_alarm = True
             delta_p = entry.price - mu
             q_release = K_P * delta_p
             alarm_msg = f"ALARM: Current Price ₹{entry.price} > UCL ₹{ucl:.2f}"
             action_msg = f"RECOMMENDATION: RELEASE {q_release:.2f} TONNES."

        return {
            "status": "success",
            "is_alarm": is_alarm,
            "stats": {
                "mu": round(mu, 2),
                "sigma": round(sigma, 2),
                "ucl": round(ucl, 2)
            },
            "forecast": [{"date": d, "price": round(p, 2)} for d, p in zip(forecast_dates, forecast_prices)],
            "message": alarm_msg,
            "action": action_msg
        }

    except Exception as e:
        print(f"Critical Error: {e}")
        # Return 500 only on catastrophic failure, but try to return JSON
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/admin/history/{location}/{commodity}")
def get_admin_history(location: str, commodity: str):
    if price_collection is None: return []
    # Get data for chart (Ascending Order)
    cursor = price_collection.find(
        {"location": location, "commodity": commodity},
        {"_id": 0, "date": 1, "price": 1}
    ).sort("date", 1).limit(7)
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