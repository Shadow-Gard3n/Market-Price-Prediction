import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import joblib
import os
import io
import base64

MODELS_DIR = 'Market_Models'
models = {}
for model_file in os.listdir(MODELS_DIR):
    if model_file.endswith('.pkl'):
        commodity_name = model_file.replace('.pkl', '').replace('_', '/')
        models[commodity_name] = joblib.load(os.path.join(MODELS_DIR, model_file))
        print(f"✅ Model loaded for: {commodity_name}")


try:
    DF_FULL = pd.read_csv('Market_Dataset/final_output.csv', parse_dates=['created_at'], index_col='created_at')
    print("✅ Dataset loaded and indexed by 'created_at'.")
except FileNotFoundError:
    print("'your_dataset.csv' not found. The application cannot start.")
    DF_FULL = None 


def _create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['price_lag_7'] = df['modal_price'].shift(7)
    df['price_lag_14'] = df['modal_price'].shift(14)
    df['price_lag_30'] = df['modal_price'].shift(30)
    df['rolling_mean_30'] = df['modal_price'].shift(1).rolling(window=30).mean()
    df['rolling_std_30'] = df['modal_price'].shift(1).rolling(window=30).std()
    return df.dropna()

def _forecast_six_months(model, df_full, commodity, last_known_date):
    df_commodity = df_full[df_full['commodity'] == commodity]
    df_daily = df_commodity.groupby(df_commodity.index).agg({'modal_price': 'mean'})
    
    future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=180, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    future_df['modal_price'] = np.nan
    df_extended = pd.concat([df_daily, future_df])
    
    for date in future_dates:
        featured_row = _create_features(df_extended.loc[:date]).iloc[-1:]
        FEATURES = [col for col in featured_row.columns if col != 'modal_price']
        prediction = model.predict(featured_row[FEATURES])[0]
        df_extended.loc[date, 'modal_price'] = prediction
    
    daily_forecast_df = df_extended.loc[future_dates].copy()
    daily_forecast_df.rename(columns={'modal_price': 'forecast'}, inplace=True)
    return daily_forecast_df

def get_market_prediction(commodity: str):

    if DF_FULL is None:
        return {"error": "Dataset not found. Please check server configuration."}
        
    if commodity not in models:
        return {"error": f"Model for commodity '{commodity}' not found."}

    model = models[commodity]

    # Prepare data for the specific commodity
    df_commodity = DF_FULL[DF_FULL['commodity'] == commodity]
    df_daily = df_commodity.groupby(df_commodity.index).agg({'modal_price': 'mean'})
    df_featured = _create_features(df_daily)
    
    test_df = df_featured.loc[df_featured.index >= '2024-01-01']
    if test_df.empty:
        return {"error": f"Not enough recent data to make a prediction for '{commodity}'."}

    FEATURES = [col for col in test_df.columns if col != 'modal_price']
    TARGET = 'modal_price'
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    predictions = model.predict(X_test)

    last_known_date = test_df.index.max()
    daily_forecast_df = _forecast_six_months(model, DF_FULL, commodity, last_known_date)
    monthly_forecast_df = daily_forecast_df.resample('ME').last().head(6)

    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(y_test.index, y_test, label='Actual Price (Recent History)', color='green', linewidth=2)
    plt.plot(y_test.index, predictions, label='Model Prediction (on Recent History)', color='red', linestyle='--')
    plt.plot(daily_forecast_df.index, daily_forecast_df['forecast'], label='6-Month Forecast', color='purple', linestyle=':')
    plt.title(f'{commodity} Price: History, Prediction & Forecast', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Modal Price')
    plt.legend()
    plt.tight_layout() 

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close() 

    formatted_forecast = monthly_forecast_df.reset_index().rename(columns={'index': 'date'}).to_dict('records')

    return {
        "commodity": commodity,
        "monthly_forecast": formatted_forecast,
        "plot_base64": plot_base64
    }