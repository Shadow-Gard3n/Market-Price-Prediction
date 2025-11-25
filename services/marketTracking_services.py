import httpx
from datetime import date, timedelta
from fastapi import HTTPException
from schemas.marketTracker_schemas import MarketPriceRequest, MarketPriceData

# The external API endpoint we are fetching data from
ENAM_API_URL = "https://enam.gov.in/web/Ajax_ctrl/trade_data_list"

async def fetch_market_data(request: MarketPriceRequest) -> list[MarketPriceData]:
    
    today = date.today()
    start_date = today - timedelta(days=6)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = today.strftime("%Y-%m-%d")

    payload = {
        "language": "en",
        "stateName": request.state_name,
        "apmcName": request.apmc_name,
        "commodityName": request.commodity_name,
        "fromDate": start_date_str,
        "toDate": end_date_str,
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(ENAM_API_URL, data=payload, headers=headers, timeout=10.0)
            response.raise_for_status() 
            
            json_data = response.json()
            api_rows = json_data.get("data", [])

            if not api_rows:
                return []

            validated_data = [MarketPriceData.model_validate(row) for row in api_rows]
            return validated_data

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            raise HTTPException(status_code=502, detail=f"Failed to communicate with eNAM portal: {exc}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail="An internal server error occurred.")


