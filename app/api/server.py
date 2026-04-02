from fastapi import FastAPI
from app.api.routes import router
from app.config import settings

app = FastAPI(title=settings.api_title)
app.include_router(router)