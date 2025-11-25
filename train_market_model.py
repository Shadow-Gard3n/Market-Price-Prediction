# needs to run only once

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class CommodityPricePredictor:
    def __init__(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.set_index('created_at')
            except (KeyError, TypeError):
                raise TypeError("DataFrame must have a DatetimeIndex or a 'created_at' column to convert.")
        self.df_full = df.copy().sort_index()
        self.models = {} # to store a trained model for each commodity
        print("✅ Predictor initialized.")

    def _create_features(self, df):
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df['weekofyear'] = df.index.isocalendar().week.astype(int)

        # Lag features (price from previous periods)
        df['price_lag_7'] = df['modal_price'].shift(7)
        df['price_lag_14'] = df['modal_price'].shift(14)
        df['price_lag_30'] = df['modal_price'].shift(30)

        # Rolling window features (trend over the last month)
        df['rolling_mean_30'] = df['modal_price'].shift(1).rolling(window=30).mean()
        df['rolling_std_30'] = df['modal_price'].shift(1).rolling(window=30).std()

        return df.dropna()

    def train(self, commodity):
        """
        Trains a new XGBoost model for a specific commodity.
        """
        print(f"--- Training model for: {commodity} ---")

        df_commodity = self.df_full[self.df_full['commodity'] == commodity]
        if df_commodity.empty:
            print(f" Warning: No data found for {commodity}. Skipping training.")
            return

        df_daily = df_commodity.groupby(df_commodity.index).agg({
            'modal_price': 'mean' # Use the average price for that day
        })

        df_featured = self._create_features(df_daily)

        train_df = df_featured.loc[df_featured.index < '2024-01-01']
        test_df = df_featured.loc[df_featured.index >= '2024-01-01']

        if test_df.empty or train_df.empty:
            print(f"⚠️ Warning: Not enough data to perform train/test split for {commodity}.")
            return

        print(f"Training data from {train_df.index.min().date()} to {train_df.index.max().date()}")
        print(f"Testing data from {test_df.index.min().date()} to {test_df.index.max().date()}")

        FEATURES = [col for col in df_featured.columns if col != 'modal_price']
        TARGET = 'modal_price'

        X_train, y_train = train_df[FEATURES], train_df[TARGET]
        X_test, y_test = test_df[FEATURES], test_df[TARGET]

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            eval_metric='mae',
            early_stopping_rounds=20
        )

        model.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False)

        self.models[commodity] = model
        print(f"✅ Model for {commodity} trained and stored.")
        self.evaluate(commodity, test_df) # Evaluate right after training

    def evaluate(self, commodity, test_df):
        if commodity not in self.models:
            print(f"❌ Error: Model for {commodity} not found. Please train it first.")
            return

        model = self.models[commodity]
        FEATURES = [col for col in test_df.columns if col != 'modal_price']
        TARGET = 'modal_price'

        X_test, y_test = test_df[FEATURES], test_df[TARGET]
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\n--- Evaluation Results for {commodity} ---")
        print(f"R-squared (R²): {r2:.3f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print("--------------------------------------")

        daily_forecast_df, monthly_forecast_df = self.forecast_six_months(commodity, test_df.index.max())

        # plt.figure(figsize=(15, 6))
        # plt.style.use('seaborn-v0_8-whitegrid')
        # plt.plot(y_test.index, y_test, label='Actual Price', color='green')
        # plt.plot(y_test.index, predictions, label='Predicted Price (on test data)', color='red', linestyle='--')
        
        # if daily_forecast_df is not None and not daily_forecast_df.empty:
        #     plt.plot(daily_forecast_df.index, daily_forecast_df['forecast'], 
        #                 label='6-Month Forecast (Daily)', color='purple', linestyle=':')
        
        # plt.title(f'{commodity} Price: Actual vs. Predicted & 6-Month Forecast', fontsize=16)
        # plt.xlabel('Date')
        # plt.ylabel('Modal Price')
        # plt.legend()
        # plt.show()
        
        if monthly_forecast_df is not None and not monthly_forecast_df.empty:
            print(f"\n--- 6-Month Forecast for {commodity} (End of Month Price) ---")
            print(monthly_forecast_df.to_string(float_format="%.2f"))
            print("---------------------------------------------------------")


    def forecast_six_months(self, commodity, last_known_date):
        
        if commodity not in self.models:
            print(f"❌ Error: Model for {commodity} not found.")
            return None, None

        model = self.models[commodity]
        df_commodity = self.df_full[self.df_full['commodity'] == commodity]
        df_daily = df_commodity.groupby(df_commodity.index).agg({'modal_price': 'mean'})
        
        future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=180, freq='D')
        future_df = pd.DataFrame(index=future_dates, columns=['modal_price'])
        df_extended = pd.concat([df_daily, future_df])
        
        for date in future_dates:
            featured_row = self._create_features(df_extended.loc[:date]).iloc[-1:]
            FEATURES = [col for col in featured_row.columns if col != 'modal_price']
            prediction = model.predict(featured_row[FEATURES])[0]
            df_extended.loc[date, 'modal_price'] = prediction
        
        daily_forecast_df = df_extended.loc[future_dates].copy()
        daily_forecast_df.rename(columns={'modal_price': 'forecast'}, inplace=True)
        monthly_forecast_df = daily_forecast_df.resample('ME').last().head(6)
        return daily_forecast_df, monthly_forecast_df

def train_and_save_models(df):
    """
    Trains a model for each commodity and saves it to a .pkl file.
    """
    predictor = CommodityPricePredictor(df)
    commodities = df['commodity'].unique()
    
    # if not os.path.exists('models'):
    #     os.makedirs('models')

    for commodity in commodities:
        predictor.train(commodity)
        if commodity in predictor.models:
            model = predictor.models[commodity]
            # joblib.dump(model, f'models/{commodity}.pkl')
            # print(f"✅ Model for {commodity} saved to models/{commodity}.pkl")
            # # Replace invalid characters for filenames
            safe_commodity_name = commodity.replace('/', '_') 
            joblib.dump(model, f'Market_Models/{safe_commodity_name}.pkl')
            print(f"✅ Model for {commodity} saved to models/{safe_commodity_name}.pkl")
            

if __name__ == '__main__':
    # Load your dataset here
    # For example:
    df_final = pd.read_csv('Market_Dataset/final_output.csv') 
    train_and_save_models(df_final)