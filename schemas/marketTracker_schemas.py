from pydantic import BaseModel, Field
from typing import Union

class MarketPriceRequest(BaseModel):
    commodity_name: str = Field(..., example="Potato")
    state_name: str = Field(..., example="UTTAR PRADESH")
    apmc_name: str = Field(..., example="AGRA")

class MarketPriceData(BaseModel):
    # We map our desired field 'date' to the API's 'created_at' field.
    date: str = Field(..., alias="created_at", example="2025-09-16")
    
    # The API sends these as strings, so we accept them as Union[str, float] for safety.
    modal_price: Union[str, float] = Field(..., alias="modal_price")
    min_price: Union[str, float] = Field(..., alias="min_price")
    max_price: Union[str, float] = Field(..., alias="max_price")
    
    # Map our fields to the API's fields
    total_arrival: str = Field(..., alias="commodity_arrivals")
    total_trade: str = Field(..., alias="commodity_traded")
    
    commodity: str = Field(..., alias="commodity")
    apmc: str = Field(..., alias="apmc")

    class Config:
        # This allows Pydantic to create the model from a dictionary
        from_attributes = True
        # This is CRITICAL: it tells Pydantic to use the 'alias' names when reading the data
        populate_by_name = True

