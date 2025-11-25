import os
import glob
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta

router = APIRouter()

# Configuration
MODELS_DIR = 'Price_Models'
DATA_ROOT = 'Price_Dataset'
PREDICTION_DAYS = 60
FUTURE_DAYS = 7

MODEL_MAPPING = {
    "Onion": "Daily_Retail_Price_of_Onion_in_All_Centres,_All_Zones_1764046936",
    "Potato": "Daily_Retail_Price_of_Potato_in_All_Centres,_All_Zones_1764046936",
    "Tomato": "Daily_Retail_Price_of_Tomato_in_All_Centres,_All_Zones_1764046936",
    "Mustard_Oil": "Daily_Retail_Price_of_Mustard_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
    "Sunflower_Oil": "Daily_Retail_Price_of_Sunflower_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
    "Soya_Oil": "Daily_Retail_Price_of_Soya_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
    "Palm_Oil": "Daily_Retail_Price_of_Palm_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
    "Groundnut_Oil": "Daily_Retail_Price_of_Groundnut_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
    "Vanaspati": "Daily_Retail_Price_of_Vanaspati_(Packed)_in_All_Centres,_All_Zones_1764046572",
    "Sugar": "Daily_Retail_Price_of_Sugar_in_All_Centres,_All_Zones_1764047124",
    "Milk": "Daily_Retail_Price_of_Milk_in_All_Centres,_All_Zones_1764047124",
    "Tea_Loose": "Daily_Retail_Price_of_Tea_Loose_in_All_Centres,_All_Zones_1764047124",
    "Gur": "Daily_Retail_Price_of_Gur_in_All_Centres,_All_Zones_1764047124",
    "Salt_Pack_Iodised": "Daily_Retail_Price_of_Salt_Pack_(Iodised)_in_All_Centres,_All_Zones_1764047124",
    "Atta": "Wheat",
    "Moong_dal": "Mooong_dal",
}

def get_data_and_generate_history(commodity_name_or_prefix, model, scaler):
    """
    Fetches data and generates predictions for the historical period (2023-Present)
    to compare 'Actual' vs 'Predicted'
    """
    variations = [commodity_name_or_prefix, commodity_name_or_prefix.replace("_", " "), commodity_name_or_prefix.split("_")[0]]
    
    files = []
    for name in variations:
        found = glob.glob(f"{DATA_ROOT}/**/{name}.csv", recursive=True)
        if found:
            files = found
            break

    if not files:
        return None, None, None, None
    
    file_path = files[0]
    
    try:
        df = pd.read_csv(file_path)
        if 'date' not in df.columns or 'value' not in df.columns: return None, None, None, None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # --- 1. Prepare Future Forecast Data ---
        if len(df) < PREDICTION_DAYS: return None, None, None, None
        last_60_values = df['value'].values[-PREDICTION_DAYS:]
        last_date = df['date'].iloc[-1]

        # --- 2. Generate Historical Predictions (Batch Processing) ---
        # We want to show 2023 onwards.
        # To predict Jan 1, 2023, we need data from Nov/Dec 2022.
        
        start_date_target = pd.Timestamp("2023-01-01")
        
        # Filter data including the buffer needed for the first prediction
        # We need PREDICTION_DAYS rows *before* start_date_target
        mask_start = df['date'] >= (start_date_target - timedelta(days=100)) # Get enough buffer
        df_processing = df.loc[mask_start].copy().reset_index(drop=True)
        
        history_actual = []
        history_predicted = []

        if len(df_processing) > PREDICTION_DAYS:
            # Scale the entire processing chunk
            values_scaled = scaler.transform(df_processing['value'].values.reshape(-1, 1))
            
            X_batch = []
            valid_dates = []
            
            # Create sliding windows
            # We start creating windows such that the *target* (i) is >= 2023-01-01
            for i in range(PREDICTION_DAYS, len(df_processing)):
                current_date = df_processing['date'].iloc[i]
                
                # Only care about dates after 2023-01-01 for the graph
                if current_date >= start_date_target:
                    window = values_scaled[i-PREDICTION_DAYS:i]
                    X_batch.append(window)
                    valid_dates.append(current_date)
                    # Store Actual for this date
                    history_actual.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'price': round(float(df_processing['value'].iloc[i]), 2)
                    })

            if X_batch:
                X_batch = np.array(X_batch)
                # Run batch prediction (Fast)
                preds_scaled = model.predict(X_batch, verbose=0)
                preds = scaler.inverse_transform(preds_scaled).flatten()
                
                # Zip dates and predictions
                for d, p in zip(valid_dates, preds):
                    history_predicted.append({
                        'date': d.strftime('%Y-%m-%d'),
                        'price': round(float(p), 2)
                    })

        return last_60_values, last_date, history_actual, history_predicted
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

@router.get("/price/predict")
async def predict_price(commodity: str = Query(..., description="Name of the commodity")):
    if not commodity: raise HTTPException(status_code=400, detail="Missing commodity")

    file_prefix = MODEL_MAPPING.get(commodity, commodity)
    model_path = os.path.join(MODELS_DIR, f"{file_prefix}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{file_prefix}_scaler.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Get Data & Generate Historical Predictions
        last_60, last_date, hist_act, hist_pred = get_data_and_generate_history(file_prefix, model, scaler)
        if last_60 is None:
             last_60, last_date, hist_act, hist_pred = get_data_and_generate_history(commodity, model, scaler)

        if last_60 is None:
            raise HTTPException(status_code=400, detail="Data not available")

        # Future Forecast (7 Days)
        input_scaled = scaler.transform(last_60.reshape(-1, 1))
        curr = input_scaled.reshape(1, PREDICTION_DAYS, 1)
        future_preds = []
        
        for _ in range(FUTURE_DAYS):
            p = model.predict(curr, verbose=0)[0]
            future_preds.append(p)
            curr = np.append(curr[:, 1:, :], [[p]], axis=1)

        future_preds = scaler.inverse_transform(future_preds).flatten()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=FUTURE_DAYS)
        
        forecast_data = [{'date': d.strftime('%Y-%m-%d'), 'price': round(float(p), 2)} for d, p in zip(future_dates, future_preds)]

        return {
            'commodity': commodity,
            'history_actual': hist_act,
            'history_predicted': hist_pred,
            'forecast': forecast_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# import os
# import glob
# import pandas as pd
# import numpy as np
# from fastapi import APIRouter, HTTPException, Query
# from tensorflow.keras.models import load_model
# import joblib
# from datetime import datetime, timedelta

# router = APIRouter()

# # Configuration
# MODELS_DIR = 'Price_Models'
# DATA_ROOT = 'Price_Dataset'
# PREDICTION_DAYS = 60
# FUTURE_DAYS = 7

# # Mapping Dictionary (Keep your mapping from the previous step here!)
# MODEL_MAPPING = {
#     "Onion": "Daily_Retail_Price_of_Onion_in_All_Centres,_All_Zones_1764046936",
#     "Potato": "Daily_Retail_Price_of_Potato_in_All_Centres,_All_Zones_1764046936",
#     "Tomato": "Daily_Retail_Price_of_Tomato_in_All_Centres,_All_Zones_1764046936",
#     "Mustard_Oil": "Daily_Retail_Price_of_Mustard_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
#     "Sunflower_Oil": "Daily_Retail_Price_of_Sunflower_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
#     "Soya_Oil": "Daily_Retail_Price_of_Soya_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
#     "Palm_Oil": "Daily_Retail_Price_of_Palm_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
#     "Groundnut_Oil": "Daily_Retail_Price_of_Groundnut_Oil_(Packed)_in_All_Centres,_All_Zones_1764046572",
#     "Vanaspati": "Daily_Retail_Price_of_Vanaspati_(Packed)_in_All_Centres,_All_Zones_1764046572",
#     "Sugar": "Daily_Retail_Price_of_Sugar_in_All_Centres,_All_Zones_1764047124",
#     "Milk": "Daily_Retail_Price_of_Milk_in_All_Centres,_All_Zones_1764047124",
#     "Tea_Loose": "Daily_Retail_Price_of_Tea_Loose_in_All_Centres,_All_Zones_1764047124",
#     "Gur": "Daily_Retail_Price_of_Gur_in_All_Centres,_All_Zones_1764047124",
#     "Salt_Pack_Iodised": "Daily_Retail_Price_of_Salt_Pack_(Iodised)_in_All_Centres,_All_Zones_1764047124",
#     "Atta": "Wheat",
#     "Moong_dal": "Mooong_dal",
# }

# def get_data_for_prediction_and_graph(commodity_name):
#     """
#     Returns:
#     1. last_60_values (for prediction)
#     2. last_date (for calculating future dates)
#     3. history_data (List of dicts for the graph: [{'date': '...', 'price': ...}])
#     """
#     search_pattern = f"{DATA_ROOT}/**/{commodity_name}.csv"
#     files = glob.glob(search_pattern, recursive=True)
    
#     if not files:
#         return None, None, None
    
#     file_path = files[0]
#     try:
#         df = pd.read_csv(file_path)
        
#         if 'date' not in df.columns or 'value' not in df.columns:
#             return None, None, None
            
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.sort_values('date')
        
#         # 1. Prepare Data for Prediction (Last 60 days)
#         last_60_values = df['value'].values[-PREDICTION_DAYS:]
#         last_date = df['date'].iloc[-1]
        
#         # 2. Prepare Data for Graph (All data after Jan 1, 2023)
#         graph_df = df[df['date'] >= '2023-01-01']
#         history_data = [
#             {'date': d.strftime('%Y-%m-%d'), 'price': round(float(p), 2)} 
#             for d, p in zip(graph_df['date'], graph_df['value'])
#         ]
        
#         return last_60_values, last_date, history_data
        
#     except Exception as e:
#         print(f"Error reading file {file_path}: {e}")
#         return None, None, None

# @router.get("/price/predict")
# async def predict_price(commodity: str = Query(..., description="Name of the commodity")):
#     if not commodity:
#         raise HTTPException(status_code=400, detail="Please provide a commodity name")

#     # Resolve Filename
#     file_prefix = MODEL_MAPPING.get(commodity, commodity)
    
#     model_path = os.path.join(MODELS_DIR, f"{file_prefix}_model.keras")
#     scaler_path = os.path.join(MODELS_DIR, f"{file_prefix}_scaler.pkl")

#     if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#         raise HTTPException(status_code=404, detail=f"Model files not found for {commodity}")

#     try:
#         # Load Resources
#         model = load_model(model_path)
#         scaler = joblib.load(scaler_path)

#         # Get Data
#         # Try simple name first, then mapped name
#         last_60_values, last_date, history_data = get_data_for_prediction_and_graph(commodity)
#         if last_60_values is None:
#              last_60_values, last_date, history_data = get_data_for_prediction_and_graph(file_prefix)

#         if last_60_values is None or len(last_60_values) < PREDICTION_DAYS:
#             raise HTTPException(status_code=400, detail="Not enough historical data")

#         # Predict
#         input_scaled = scaler.transform(last_60_values.reshape(-1, 1))
#         current_batch = input_scaled.reshape(1, PREDICTION_DAYS, 1)

#         future_predictions = []
#         for i in range(FUTURE_DAYS):
#             current_pred = model.predict(current_batch, verbose=0)[0]
#             future_predictions.append(current_pred)
#             current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

#         future_predictions = scaler.inverse_transform(future_predictions).flatten()
        
#         future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=FUTURE_DAYS)
        
#         forecast_data = []
#         for date, price in zip(future_dates, future_predictions):
#             forecast_data.append({
#                 'date': date.strftime('%Y-%m-%d'),
#                 'price': round(float(price), 2)
#             })

#         return {
#             'commodity': commodity,
#             'history': history_data, # NEW: sending history
#             'forecast': forecast_data
#         }

#     except Exception as e:
#         print(f"Server Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))