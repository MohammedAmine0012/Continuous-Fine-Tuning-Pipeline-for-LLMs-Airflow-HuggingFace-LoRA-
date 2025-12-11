from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

TRITON_URL = os.environ.get("TRITON_HTTP_URL", "http://localhost:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL_NAME", "deepseek_merged")

app = FastAPI(title="DeepSeek Chat API")

# Enable CORS for local development and simple deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatTurn(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] | None = None
    temperature: float | None = None
    max_new_tokens: int | None = None

class ChatResponse(BaseModel):
    reply: str

# Serve static web UI from / (root)
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "webui")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Build a conversational prompt using simple templating
    system_preamble = (
        "You are DeepSeek, a helpful AI assistant. Respond clearly and concisely.\n"
    )

    history = req.history or []
    turns = []
    for t in history[-6:]:  # keep last 6 turns
        role = "User" if t.role == "user" else "Assistant"
        turns.append(f"{role}: {t.content}")
    turns.append(f"User: {req.message}")
    turns.append("Assistant:")
    full_prompt = system_preamble + "\n".join(turns)

    # Optionally hint generation params in prompt (Triton model currently fixed)
    if req.temperature is not None:
        full_prompt = f"[temperature={req.temperature}]\n" + full_prompt
    if req.max_new_tokens is not None:
        full_prompt = f"[max_new_tokens={req.max_new_tokens}]\n" + full_prompt

    payload = {
        "inputs": [
            {
                "name": "prompt",
                "datatype": "BYTES",
                "shape": [1, 1],
                "data": [full_prompt],
            }
        ],
        "outputs": [{"name": "text"}],
    }
    url = f"{TRITON_URL}/v2/models/{TRITON_MODEL}/infer"
    r = requests.post(url, json=payload, timeout=300)
    if not r.ok:
        # surface Triton error to client
        return ChatResponse(reply=f"[Triton error {r.status_code}] {r.text}")
    out = r.json()
    text = out.get("outputs", [{}])[0].get("data", [""])[0]
    return ChatResponse(reply=text)
