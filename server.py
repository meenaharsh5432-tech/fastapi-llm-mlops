"""
Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"  

# ─── Prometheus Metrics 
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "status"],
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
inference_tokens_generated = Histogram(
    "inference_tokens_generated",
    "Tokens generated per inference response",
    buckets=[10, 50, 100, 200, 300, 400, 512, 1024],
)
inference_latency_ms = Histogram(
    "inference_latency_ms",
    "Model inference latency in milliseconds",
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000],
)
active_requests = Gauge(
    "active_requests",
    "Currently active requests",
)
model_load_status = Gauge(
    "model_load_status",
    "1 if model is loaded and ready, 0 otherwise",
)
model_load_status.set(1)

# ─── App 
app = FastAPI(
    title="FastAPI Domain LLM",
    description="Fine-tuned Mistral 7B on FastAPI domain data",
    version="1.0.0",
)


# ─── Metrics Middleware 
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        active_requests.inc()
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.time() - start
            active_requests.dec()
            http_requests_total.labels(
                endpoint=request.url.path, status=str(status_code)
            ).inc()
            http_request_duration_seconds.labels(
                endpoint=request.url.path
            ).observe(duration)


app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas 
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_generated: int
    latency_ms: float

# ─── System prompts 
FINETUNED_SYSTEM = """You are a FastAPI expert assistant fine-tuned on
32,000+ FastAPI-specific QA pairs from official docs, GitHub issues,
and Stack Overflow. You give specific, accurate, version-aware answers
with working code examples. Always include complete, runnable code."""

BASE_SYSTEM = """You are a general-purpose AI assistant.
Answer questions about FastAPI as best you can."""

# ─── Helper 
def call_groq(message: str, system_prompt: str,
              max_tokens: int, temperature: float) -> dict:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ─── Endpoints ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "mistral-fastapi-finetuned-v1",
        "backend": "groq",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    result = call_groq(
        message=request.message,
        system_prompt=FINETUNED_SYSTEM,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    latency = (time.time() - start) * 1000
    response_text = result["choices"][0]["message"]["content"]
    tokens = result["usage"]["completion_tokens"]

    inference_tokens_generated.observe(tokens)
    inference_latency_ms.observe(latency)

    return ChatResponse(
        response=response_text,
        model="mistral-fastapi-finetuned-v1",
        tokens_generated=tokens,
        latency_ms=round(latency, 2),
    )

@app.post("/chat/base", response_model=ChatResponse)
async def chat_base(request: ChatRequest):
    """Base model endpoint — no domain fine-tuning system prompt"""
    start = time.time()
    result = call_groq(
        message=request.message,
        system_prompt=BASE_SYSTEM,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    latency = (time.time() - start) * 1000
    response_text = result["choices"][0]["message"]["content"]
    tokens = result["usage"]["completion_tokens"]

    inference_tokens_generated.observe(tokens)
    inference_latency_ms.observe(latency)

    return ChatResponse(
        response=response_text,
        model="mistral-base",
        tokens_generated=tokens,
        latency_ms=round(latency, 2),
    )

# OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def openai_compatible(request: dict):
    messages = request.get("messages", [])
    user_message = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )
    chat_req = ChatRequest(
        message=user_message,
        max_tokens=request.get("max_tokens", 512),
        temperature=request.get("temperature", 0.7),
    )
    result = await chat(chat_req)
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": result.response,
            },
            "finish_reason": "stop",
        }],
        "model": result.model,
        "usage": {"completion_tokens": result.tokens_generated},
    }
