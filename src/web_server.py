from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import requests
import os
import logging
import sys
import uvicorn
import socket
import datetime
from typing import List, Dict, Any, Optional
import torch

# Import local chat module for fallback
from .chat import load_inference_model

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
TRITON_URL = os.environ.get("TRITON_HTTP_URL", "http://localhost:8000")
TRITON_MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "deepseek_merged")
PORT = int(os.environ.get("PORT", 8080))

# Global model cache for local mode
LOCAL_MODEL = None
LOCAL_TOKENIZER = None

# Test if port is available
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Lifespan handler for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOCAL_MODEL, LOCAL_TOKENIZER
    # Startup
    logger.info("="*50)
    logger.info("Starting StackOverflow Assistant API")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Triton URL: {TRITON_URL}")
    logger.info(f"Server will run on port: {PORT}")
    
    # Test port availability
    if is_port_in_use(PORT):
        logger.warning(f"Port {PORT} is already in use!")
    
    # Test Triton connection
    app.state.local_mode = False
    app.state.mock_mode = False
    
    try:
        logger.info("Testing Triton server connection...")
        response = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=5)
        logger.info(f"Triton server status: {response.status_code} - {response.text.strip()}")
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {e}")
        logger.warning("!!! TRITON IS UNREACHABLE - ATTEMPTING LOCAL MODEL LOAD !!!")
        try:
            logger.info("Loading model locally... (this may take a minute)")
            # Load base model + adapters
            LOCAL_MODEL, LOCAL_TOKENIZER = load_inference_model()
            app.state.local_mode = True
            logger.info("Local model loaded successfully!")
        except Exception as load_err:
            logger.error(f"Failed to load local model: {load_err}")
            logger.warning("Falling back to MOCK mode due to load failure.")
            app.state.mock_mode = True 

    yield  # This is where the application runs
    
    # Shutdown
    logger.info("Shutting down StackOverflow Assistant API")

app = FastAPI(
    title="StackOverflow Assistant API",
    description="API for interacting with DeepSeek model via Triton Inference Server or Local Fallback",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS
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

# Serve static web UI
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "webui")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "OK"

@app.get("/triton/health")
async def triton_health():
    try:
        response = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=5)
        return {
            "triton_status": "online" if response.status_code == 200 else f"error: {response.status_code}",
            "triton_response": response.text.strip()
        }
    except Exception as e:
        return {
            "triton_status": "error",
            "message": str(e)
        }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info(f"Received chat request with message: {req.message[:100]}...")
    
    try:
        # 1. Local/Mock Mode
        if getattr(req, "app", None):
            # Mock Mode
            if getattr(req.app.state, "mock_mode", False):
                 import asyncio
                 await asyncio.sleep(1.0)
                 return ChatResponse(reply=f"**[MOCK MODE]**\n\nDocker is down and local model failed to load.\n\nSimulated response.")
            
            # Local Inference Mode
            if getattr(req.app.state, "local_mode", False) and LOCAL_MODEL and LOCAL_TOKENIZER:
                 logger.info("Generating response using LOCAL model...")
                 
                 system_msg = "You are StackOverflow Assistant, an expert programming AI. Respond appropriately to code questions."
                 prompt_parts = [system_msg]
                 history = req.history or []
                 for t in history[-6:]:
                     role_prefix = "### Instruction" if t.role == "user" else "### Response"
                     prompt_parts.append(f"{role_prefix}:\n{t.content}")
                 prompt_parts.append(f"### Instruction:\n{req.message}")
                 prompt_parts.append("### Response:\n")
                 full_prompt = "\n\n".join(prompt_parts)
                 
                 inputs = LOCAL_TOKENIZER(full_prompt, return_tensors="pt").to(LOCAL_MODEL.device)
                 
                 with torch.no_grad():
                     outputs = LOCAL_MODEL.generate(
                         **inputs,
                         max_new_tokens=req.max_new_tokens or 512,
                         temperature=req.temperature or 0.7,
                         do_sample=True,
                         top_p=0.9,
                         repetition_penalty=1.1,
                         pad_token_id=LOCAL_TOKENIZER.eos_token_id
                     )
                 response_text = LOCAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
                 
                 if "### Response:" in response_text:
                     answer = response_text.split("### Response:")[-1].strip()
                 else:
                     answer = response_text
                     
                 # Stop tokens
                 stop_tokens = ["### Instruction:", "User:"]
                 for token in stop_tokens:
                     if token in answer:
                         answer = answer.split(token)[0].strip()
                         
                 return ChatResponse(reply=answer)

        # 2. Triton Mode
        system_msg = "You are StackOverflow Assistant, an expert programming AI. Respond appropriately to code questions."
        prompt_parts = [system_msg]
        history = req.history or []
        for t in history[-6:]:
            role_prefix = "### Instruction" if t.role == "user" else "### Response"
            prompt_parts.append(f"{role_prefix}:\n{t.content}")
        prompt_parts.append(f"### Instruction:\n{req.message}")
        prompt_parts.append("### Response:\n")
        full_prompt = "\n\n".join(prompt_parts)

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
        
        url = f"{TRITON_URL}/v2/models/{TRITON_MODEL_NAME}/infer"
        logger.info(f"Sending request to Triton at {url}")
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            if not response.ok:
                error_msg = f"Triton error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return ChatResponse(reply=f"[Error] {error_msg}")
                
            out = response.json()
            text = out.get("outputs", [{}])[0].get("data", [""])[0]
            
            stop_tokens = ["### Instruction:", "User:"]
            
            # If the model echoes the prompt (which it often does), we need to extract the answer.
            if "### Response:" in text:
                text = text.split("### Response:")[-1].strip()
            
            for token in stop_tokens:
                if token in text:
                    text = text.split(token)[0].strip()
            
            return ChatResponse(reply=text)
            
        except requests.exceptions.RequestException as e:
            return ChatResponse(reply=f"[Error] {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return ChatResponse(reply=f"[Error] {str(e)}")
