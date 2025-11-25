from pydantic import BaseModel
from typing import Dict

class MarketPriceInput(BaseModel):
    commodity: str

class MarketPriceOutput(BaseModel):
    commodity: str
    forecast: Dict[str, float]
