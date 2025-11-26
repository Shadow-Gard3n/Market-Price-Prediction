# Market Price Prediction System

A comprehensive Machine Learning application designed to predict the market prices of various commodities (Vegetables, Grains, Oils, etc.) based on historical data from Mandis. This system uses an XGBoost regression model served via a FastAPI backend and visualizes the data through a web-based frontend.

## 🚀 Features

* **Price Prediction:** Predicts the modal price of commodities for upcoming dates using XGBoost.
* **6-Month Forecast:** Generates a daily and monthly forecast trend for specific crops.
* **Market Tracker:** Interactive frontend to view current trends and historical data.
* **Admin Dashboard:** Functionality for managing data and system configurations.
* **Dynamic Data Mapping:** Automatically maps States to APMCs (Agricultural Produce Market Committees) for easy user selection.
* **API Support:** Robust REST API built with FastAPI including CORS support for local development.

## 🛠️ Tech Stack

### Backend
* **Framework:** Python, FastAPI
* **Server:** Uvicorn
* **Database:** MongoDB (via `pymongo`)

### Machine Learning & Data
* **Core Model:** XGBoost (Regressor)
* **Data Manipulation:** Pandas, NumPy
* **Utilities:** Joblib (Model persistence), Scikit-Learn (Metrics)
* **Visualization:** Matplotlib

### Frontend
* **Core:** HTML5, CSS3, JavaScript (Vanilla)

## 📂 Folder Structure

```text
root/
├── frontend/                  # Client-side application
│   ├── css/                   # Stylesheets (market.css, price.css)
│   ├── admin.html             # Admin dashboard interface
│   ├── index.html             # Landing page
│   ├── market.html            # Market trends view
│   ├── price.html             # Price prediction interface
│   └── apmc_data.json         # Generated mapping of States to APMCs
│
├── notebooks/                 # Jupyter notebooks for analysis & processing
│   ├── Center_Price_Prediction.ipynb
│   ├── Cleaning_Dataset.ipynb # Data cleaning logic
│   ├── Json_Data.ipynb        # Script to generate apmc_data.json
│   ├── Market_Price_Final.ipynb
│   └── ...
│
├── price_dataset/             # Raw CSV Data (from Agmarknet/eNam)
│   ├── EdibleOils/
│   ├── Grains/
│   ├── Pulses/
│   ├── Vegetables/
│   └── ...
│
├── routes/                    # FastAPI Route Controllers
│   ├── admin.py
│   ├── prediction.py
│   ├── price_prediction.py
│   └── ...
│
├── schemas/                   # Pydantic Models for Data Validation
├── services/                  # Database and Logic Services
├── Market_Models/             # Directory where .pkl models are saved
├── main.py                    # Application Entry Point
├── train_market_model.py      # Script to train and save XGBoost models
└── requirements.txt           # Python Dependencies
```


## ⚙️ Installation & Setup Process

Follow these steps to set up the project locally.

### 1. Prerequisites
Ensure you have Python 3.8+ and MongoDB installed on your system.

### 2. Install Dependencies
Navigate to the project root and install the required Python packages:

```
pip install -r requirements.txt
```

### 3. Data Acquisition & Preparation
The system relies on historical price data (CSV format) from sources like **Agmarknet** or **eNam Mandis**.

1.  **Download Data:** Place your raw CSV files (e.g., `Daily Retail Price of Onion...`) into the `price_dataset/` folder, organized by category (Vegetables, Grains, etc.).
2.  **Clean Data:** Run the cleaning notebook to merge and format the raw CSVs into a single dataset for training.
    * Open `notebooks/Cleaning_Dataset.ipynb` or `notebooks/Market_Price_Final.ipynb`.
    * Run the cells to produce `Market_Dataset/final_output.csv`.

### 4. Generate Frontend Data
To ensure the dropdown menus in the frontend work correctly (State -> APMC mapping), you must run the JSON generation script.

1.  Open `notebooks/Json_Data.ipynb`.
2.  Run the notebook.
3.  **Output:** This will create/update `frontend/apmc_data.json`.

### 5. Train the Models
You need to train the XGBoost models before making predictions. Run the training script:
```
python train_market_model.py
```

* **Input:** Reads `Market_Dataset/final_output.csv` (ensure this exists from Step 3).
* **Output:** Saves trained models (`.pkl` files) into the `Market_Models/` directory.

### 6. Start the Backend Server
Launch the FastAPI application:
```
uvicorn main:app --reload
```

* The API will be available at `http://127.0.0.1:8000`.
* Swagger UI docs are available at `http://127.0.0.1:8000/docs`.

### 7. Run the Frontend
You can serve the `frontend` folder using any static file server (like VS Code "Live Server") or simply open `frontend/index.html` in your browser.

* Ensure the frontend is running on port `5500` or `5501` (as configured in `main.py` CORS settings).

## 🔮 Usage

1.  **Go to the Price Prediction Page:** Select a commodity, state, and market.
2.  **View Forecasts:** The system will query the backend, load the specific `.pkl` model, and return the predicted price and 6-month forecast.