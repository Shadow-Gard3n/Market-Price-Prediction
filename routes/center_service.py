# import os
# import joblib
# import pandas as pd
# import numpy as np
# import glob
# from datetime import datetime
# from fastapi import APIRouter, HTTPException, Query
# from tensorflow.keras.models import load_model
# from services.database import price_collection

# router = APIRouter()

# # Configuration
# MODELS_ROOT = 'Center_Price_Models'
# DATASET_ROOT = 'Center_Price_Dataset'
# PREDICTION_DAYS = 30

# # Mapping Dropdown values to Folder/File names
# # Keys = Dropdown Value, Values = Folder Name (Capitalized)
# COMMODITY_MAP = {
#     "Groundnut_oil": "Groundnut",
#     "Mustard_oil": "Mustard",
#     "Vanaspati_oil": "Vanaspati",
#     "Soya_oil": "Soya",
#     "Palm_oil": "Palm",
#     "Sunflower_oil": "Sunflower",
#     "Tea_loose": "Tea",
#     "Tur_dal": "Toor",
#     "Moong_dal": "Moong",
#     "Urad_dal": "Urad",
#     "Masoor_dal": "Masoor",
#     "Gram_dal": "Gram",
#     "Salt": "Salt",
#     "Gur": "Gur",
#     "Sugar": "Sugar",
#     "Milk": "Milk",
#     "Atta": "Atta",
#     "Rice": "Rice",
#     "Wheat": "Wheat",
#     "Onion": "Onion",
#     "Potato": "Potato",
#     "Tomato": "Tomato"
# }

# def get_last_training_data(commodity_folder):
#     """
#     Finds the CSV for the commodity to get the last 30 days of 'actual' data.
#     """
#     # Try to find the file dynamically
#     search_pattern = f"{DATASET_ROOT}/*{commodity_folder.lower()}*.csv"
#     files = glob.glob(search_pattern)
    
#     if not files:
#         return None, None
    
#     df = pd.read_csv(files[0])
#     df.columns = [c.lower().strip() for c in df.columns]
    
#     if 'date' not in df.columns or 'price' not in df.columns:
#         return None, None
        
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values('date')
    
#     # Return the last valid date and the dataframe
#     return df['date'].iloc[-1], df

# @router.get("/api/center/predict")
# async def predict_center_price(center: str, commodity: str, target_date: str):
#     # 1. Resolve Commodity Name
#     folder_name = COMMODITY_MAP.get(commodity, commodity.capitalize())
    
#     # 2. Paths
#     model_path = os.path.join(MODELS_ROOT, folder_name, center, 'model.keras')
#     scaler_path = os.path.join(MODELS_ROOT, folder_name, center, 'scaler.pkl')

#     if not os.path.exists(model_path):
#         return {"status": "error", "message": f"No model found for {center} - {commodity}"}

#     try:
#         # 3. Load Resources
#         model = load_model(model_path)
#         scaler = joblib.load(scaler_path)

#         # 4. Get Seed Data
#         last_date, df = get_last_training_data(folder_name)
#         if df is None:
#             return {"status": "error", "message": "Dataset not found."}

#         df_center = df[df['centre_name'] == center].sort_values('date')
#         if len(df_center) < PREDICTION_DAYS:
#             return {"status": "error", "message": "Insufficient history."}

#         # Prepare seed sequence
#         last_sequence = df_center['price'].values[-PREDICTION_DAYS:].reshape(-1, 1)
#         current_batch = scaler.transform(last_sequence).reshape(1, PREDICTION_DAYS, 1)

#         # 5. Calculate Gap
#         target_dt = pd.to_datetime(target_date)
#         days_gap = (target_dt - last_date).days

#         if days_gap < 1:
#              return {"status": "error", "message": "Date is in the past relative to training data."}

#         # 6. Recursive Prediction Loop (Gap + 6 extra days = 7 days total)
#         forecast_results = []
#         total_iterations = days_gap + 6

#         for i in range(total_iterations):
#             pred = model.predict(current_batch, verbose=0)[0]
            
#             # Update batch for next step
#             current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

#             # Only store if we have reached the target start date
#             # i=0 corresponds to (last_date + 1 day)
#             current_step_date = last_date + pd.Timedelta(days=i+1)

#             if current_step_date >= target_dt:
#                 price_val = scaler.inverse_transform([[pred[0]]])[0][0]
#                 forecast_results.append({
#                     "date": current_step_date.strftime("%Y-%m-%d"),
#                     "price": round(float(price_val), 2)
#                 })

#         return {
#             "status": "success",
#             "forecast": forecast_results
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# @router.get("/api/center/history")
# async def get_center_history(center: str, commodity: str):
#     """
#     Fetches the last 7 entries entered by the Admin for this center.
#     """
#     if price_collection is None:
#         return []

#     cursor = price_collection.find(
#         {"location": center, "commodity": commodity},
#         {"_id": 0, "date": 1, "price": 1}
#     ).sort("date", -1).limit(7) # Sort descending to get latest
    
#     data = list(cursor)
    
#     if not data:
#         return {"status": "empty", "message": "No data found for this center."}
        
#     return {"status": "success", "data": data}


import os
import joblib
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from fastapi import APIRouter
from tensorflow.keras.models import load_model
from services.database import price_collection

router = APIRouter()

MODELS_ROOT = 'models'
DATASET_ROOT = 'Center_Price_Dataset'
PREDICTION_DAYS = 30

COMMODITY_MAP = {
    "Groundnut_oil": "Groundnut", "Mustard_oil": "Mustard", "Vanaspati_oil": "Vanaspati",
    "Soya_oil": "Soya", "Palm_oil": "Palm", "Sunflower_oil": "Sunflower",
    "Tea_loose": "Tea", "Toor": "Toor", "Moong": "Moong", "Urad": "Urad",
    "Masoor": "Masoor", "Gram": "Gram", "Salt": "Salt", "Gur": "Gur",
    "Sugar": "Sugar", "Milk": "Milk", "Atta": "Atta", "Rice": "Rice",
    "Wheat": "Wheat", "Onion": "Onion", "Potato": "Potato", "Tomato": "Tomato"
}

def get_last_training_data(commodity_folder):
    search_pattern = f"{DATASET_ROOT}/*{commodity_folder.lower()}*.csv"
    files = glob.glob(search_pattern)
    if not files: return None, None
    df = pd.read_csv(files[0])
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df['date'].iloc[-1], df

@router.get("/api/center/predict")
async def predict_center_price(center: str, commodity: str, target_date: str):
    folder_name = COMMODITY_MAP.get(commodity, commodity.capitalize())
    model_path = os.path.join(MODELS_ROOT, folder_name, center, 'model.keras')
    scaler_path = os.path.join(MODELS_ROOT, folder_name, center, 'scaler.pkl')

    if not os.path.exists(model_path):
        return {"status": "warning", "message": "No model found for this Center/Commodity."}

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        last_date, df = get_last_training_data(folder_name)
        
        if df is None:
            return {"status": "warning", "message": "Historical dataset missing."}

        df_center = df[df['centre_name'] == center].sort_values('date')
        
        if len(df_center) < PREDICTION_DAYS:
            return {"status": "warning", "message": "Not enough data to predict."}

        last_sequence = df_center['price'].values[-PREDICTION_DAYS:].reshape(-1, 1)
        current_batch = scaler.transform(last_sequence).reshape(1, PREDICTION_DAYS, 1)

        target_dt = pd.to_datetime(target_date)
        days_gap = (target_dt - last_date).days

        if days_gap < 1:
             return {"status": "warning", "message": "Cannot predict for past dates. Select a future date."}

        forecast_results = []
        total_iterations = days_gap + 6 

        for i in range(total_iterations):
            pred = model.predict(current_batch, verbose=0)[0]
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
            
            current_step_date = last_date + pd.Timedelta(days=i+1)
            
            if current_step_date >= target_dt:
                price_val = scaler.inverse_transform([[pred[0]]])[0][0]
                forecast_results.append({
                    "date": current_step_date.strftime("%Y-%m-%d"),
                    "price": round(float(price_val), 2)
                })

        return {"status": "success", "forecast": forecast_results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/api/center/history")
async def get_center_history(center: str, commodity: str):
    if price_collection is None: return {"status": "error", "data": []}
    
    # Fetch last 7 by date descending
    cursor = price_collection.find(
        {"location": center, "commodity": commodity},
        {"_id": 0, "date": 1, "price": 1}
    ).sort("date", -1).limit(7)
    
    return {"status": "success", "data": list(cursor)}