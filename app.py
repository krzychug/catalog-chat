import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catalog_logic import init_catalog, chat_with_session, reset_session

CATALOG_READY = False


async def _background_init():
    global CATALOG_READY
    try:
        print("[startup] Initializing catalog in background...")
        await asyncio.get_event_loop().run_in_executor(None, init_catalog)
        CATALOG_READY = True
        print("[startup] Catalog ready.")
    except Exception as e:
        print(f"[startup] ERROR during catalog init: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire-and-forget: port binds immediately, catalog loads in background
    asyncio.create_task(_background_init())
    yield


app = FastAPI(title="DABSTORY Catalog Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/")
def root():
    if os.path.isfile("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse({"status": "running", "info": "no frontend found"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"catalog_ready": CATALOG_READY}


@app.post("/chat")
def chat(req: ChatRequest):
    if not CATALOG_READY:
        return JSONResponse(
            status_code=503,
            content={"answer": "Katalog jest w trakcie ładowania, proszę czekać chwilę i spróbuj ponownie."},
        )
    answer = chat_with_session(req.session_id, req.message)
    return {"answer": answer}


@app.post("/reset")
def reset(req: ResetRequest):
    reset_session(req.session_id)
    return {"status": "reset"}
