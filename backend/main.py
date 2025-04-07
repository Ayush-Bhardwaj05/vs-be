from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
# from routes.predict import router as predict_router
from routes.convo import convo_router
from routes.auth_routes import router as auth_router, initialize_collection
from auth import verify_token
from motor.motor_asyncio import AsyncIOMotorClient
from routes.predict import router as predict_router
from routes.predict_indian import router as predict2_router
from routes.emotion_route import router as emotion_router
from routes.video_routes import router as video_router
app = FastAPI()

# Add CORS middleware configuration
origins = [
    "*",  # adjust this if you have different frontend URLs
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or allow_origins=["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(predict_router)
app.include_router(convo_router)
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(predict2_router)
app.include_router(emotion_router)
app.include_router(video_router)
DATABASE_URL = "mongodb+srv://harshdaftari:harshdaftari123@cluster0.cz6dg.mongodb.net/"
DATABASE_NAME = "my_database"
COLLECTION_NAME = "my_collection"

# Initialize MongoDB
client = None
db = None
collection = None

async def init_db():
    global client, db, collection
    try:
        client = AsyncIOMotorClient(DATABASE_URL)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        initialize_collection(collection)
        print("✅ Database connection established successfully.")
    except Exception as e:
        print(f"❗ Error connecting to database: {e}")

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()
        print("❎ Database connection closed.")

@app.get("/")
def root():
    return {"message": "Welcome to the Speech Emotion Recognition API!"}

@app.get("/protected")
def protected_route(username: str = Depends(verify_token)):
    return {"message": f"Welcome, {username}! This is a protected route."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
