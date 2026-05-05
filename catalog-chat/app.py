import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catalog_logic import init_catalog, chat_with_session, reset_session

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_catalog()
    yield

app = FastAPI(title="DABSTORY Catalog Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ResetRequest(BaseModel):
    session_id: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    answer = chat_with_session(req.session_id, req.message)
    return {"answer": answer}

@app.post("/reset")
def reset(req: ResetRequest):
    reset_session(req.session_id)
    return {"status": "reset"}
