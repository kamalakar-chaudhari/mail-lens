from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import router as chat_router

load_dotenv()
app = FastAPI()

# Register chat routes
app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "Welcome to the AI Agent API"}
