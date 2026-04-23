# FastAPI Domain-Specific LLM — Full MLOps Pipeline

A production-ready LLM inference server built with FastAPI, backed by a Mistral 7B model fine-tuned on 32,000+ FastAPI-specific QA pairs. The stack includes full observability (Prometheus + Grafana), a Streamlit comparison UI, CI/CD via GitHub Actions, and Docker Compose orchestration.

---

## Architecture

```
User / Streamlit UI (port 8501)
        |
        v
  FastAPI Server (port 8000)
  ├── POST /chat          → Fine-tuned system prompt → Groq API
  ├── POST /chat/base     → Base system prompt       → Groq API
  ├── GET  /metrics       → Prometheus scrape
  └── GET  /health
        |
        v
  Prometheus (port 9090)  ←──── scrapes /metrics every 5s
        |
        v
  Grafana (port 3000)     ←──── visualises Prometheus data
```

---

## Training Results

| Metric | Value |
|--------|-------|
| Base model | Mistral 7B |
| Eval loss | 0.97 |
| Epochs | 1.2 |
| Dataset size | 32,000+ QA pairs |
| W&B run | https://wandb.ai |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Inference API | FastAPI + Uvicorn |
| LLM backend | Groq API (LLaMA 3.3 70B) |
| Fine-tuning | LoRA / PEFT on Mistral 7B |
| Metrics | Prometheus + prometheus-client |
| Dashboards | Grafana |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## How to Run Locally

```bash
pip install -r requirements.txt

# copy and fill in your Groq key
cp .env.example .env

uvicorn server:app --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs

# optional: Streamlit comparison UI
streamlit run app.py
```

---

## How to Run with Docker

```bash
docker-compose up --build
```

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI | http://localhost:8000/docs | — |
| Metrics | http://localhost:8000/metrics | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

```bash
# stop stack
docker-compose down

# stop and wipe volumes
docker-compose down -v
```

---

## Monitoring

Prometheus scrapes `/metrics` every 5 seconds. Grafana auto-loads the **FastAPI LLM Monitoring** dashboard on startup.

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total requests by endpoint and status |
| `http_request_duration_seconds` | Histogram | Request latency per endpoint |
| `inference_tokens_generated` | Histogram | Tokens returned per inference call |
| `inference_latency_ms` | Histogram | End-to-end model call latency |
| `active_requests` | Gauge | In-flight requests |
| `model_load_status` | Gauge | 1 = ready, 0 = not ready |

Dashboard panels: Request Rate, P99 Latency, Tokens Generated, Active Requests.

---

## Dataset & Model

- **Dataset:** FastAPI-specific QA pairs scraped from official docs, GitHub issues, and Stack Overflow
- **Base model:** `mistralai/Mistral-7B-v0.1`
- **Adapter:** LoRA weights saved in `model/`

---

## Resume Bullets

- Fine-tuned Mistral 7B on a custom 32K-sample FastAPI QA dataset using LoRA/PEFT, achieving eval loss of 0.97
- Built a production FastAPI inference server with Prometheus metrics, Grafana dashboards, and a Streamlit comparison UI
- Containerised the full stack (FastAPI + Prometheus + Grafana) with Docker Compose and automated CI/CD via GitHub Actions
- Implemented OpenAI-compatible `/v1/chat/completions` endpoint for drop-in API compatibility
