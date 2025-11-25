import os
import urllib.parse
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Fetch variables
# We use 'raw' strings from .env and encode them safely here
raw_username = os.getenv("MONGO_USER")
raw_password = os.getenv("MONGO_PASSWORD")
cluster_url = os.getenv("MONGO_CLUSTER")

# Initialize variables
client = None
db = None
price_collection = None

# Check if credentials exist to avoid errors
if not raw_username or not raw_password:
    print("❌ Error: MongoDB credentials not found in .env file.")
else:
    try:
        # 3. URL Encode credentials (handles special chars like '@')
        encoded_username = urllib.parse.quote_plus(raw_username)
        encoded_password = urllib.parse.quote_plus(raw_password)

        # 4. Construct URI
        MONGO_URI = f"mongodb+srv://{encoded_username}:{encoded_password}@{cluster_url}/?appName=Cluster0"

        # 5. Connect
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')  # Test connection

        db = client['agri_market_db']
        price_collection = db['price_entries']
        print("✅ Connected to MongoDB Successfully")

    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")