from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, export, health, projects, ws
from app.core.config import get_settings
from app.core.database import ensure_indexes
from app.core.storage import ensure_bucket

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(projects.router)
app.include_router(export.router)
app.include_router(chat.router)
app.include_router(ws.router)


@app.on_event("startup")
async def on_startup():
    await ensure_indexes()
    ensure_bucket()
