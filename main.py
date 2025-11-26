from fastapi import FastAPI
from routes import prediction
from routes import admin
from routes import price_prediction
from routes import center_service


from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://127.0.0.1:5501", 
    "http://localhost:5501",   
    "http://127.0.0.1:5500", 
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],    
)

app.include_router(prediction.router)
app.include_router(admin.router)
app.include_router(price_prediction.router)
app.include_router(center_service.router)