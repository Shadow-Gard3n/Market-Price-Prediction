from fastapi import FastAPI
from routes import prediction
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://127.0.0.1:5501", # Your frontend's origin
    "http://localhost:5501",   # Also a good idea to include localhost
    "http://127.0.0.1:5500", # Your frontend's origin
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all headers
)

app.include_router(prediction.router)