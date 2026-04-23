
# FastAPI Domain-Specific LLM — Full MLOps Pipeline

![CI](https://github.com/meenaharsh5432-tech/fastapi-llm-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

A QLoRA fine-tuned Mistral 7B model served through a production-grade FastAPI inference server. The project covers the full MLOps lifecycle: dataset curation (32,575 FastAPI-specific QA pairs), 4-bit quantised training with PEFT/LoRA, a Groq-backed inference API with OpenAI-compatible endpoints, Prometheus + Grafana observability, a Streamlit side-by-side comparison UI, Docker Compose orchestration, and automated CI/CD via GitHub Actions.

---

## Architecture

```
  Data Sources                Pipeline               Training
  ────────────                ────────               ────────
  Official Docs  ──┐                                 QLoRA 4-bit
  GitHub Issues  ──┼──► scrape ──► clean ──► push ──► Mistral 7B
  Stack Overflow ──┘    32,575 pairs to HF Hub        1.2 epochs
                                                       eval loss 0.97
                                                            │
                         Serving                           ▼
                         ───────               fine-tuned adapter
  Streamlit UI (8501) ◄──┐                         (HF Hub)
                          │                              │
                    FastAPI (8000)  ◄────────────────────┘
                    ├── POST /chat          Groq API (LLaMA 3.3 70B)
                    ├── POST /chat/base     + fine-tuned system prompt
                    ├── POST /v1/chat/completions  (OpenAI-compatible)
                    ├── GET  /health
                    └── GET  /metrics
                                │
                         Monitoring
                         ──────────
                    Prometheus (9090) ◄── scrapes /metrics every 5s
                         │
                    Grafana (3000) ──► Request Rate, P99 Latency,
                                       Tokens Generated, Active Requests
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Training pairs | 32,575 |
| Base model | Mistral 7B (`mistralai/Mistral-7B-v0.1`) |
| Fine-tuning method | QLoRA 4-bit (PEFT / LoRA) |
| Epochs | 1.2 |
| Eval loss | 0.97 |
| Inference latency | ~2s per response (via Groq) |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Inference API | FastAPI + Uvicorn |
| LLM backend | Groq API (LLaMA 3.3 70B) |
| Fine-tuning | QLoRA 4-bit, PEFT / LoRA on Mistral 7B |
| Dataset | 32,575 FastAPI QA pairs, pushed to HF Hub |
| Experiment tracking | Weights & Biases |
| Metrics | Prometheus + prometheus-client |
| Dashboards | Grafana |
| UI | Streamlit (fine-tuned vs base comparison) |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions (test, lint, docker-build, notify) |

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env   # then fill in GROQ_API_KEY

# 3. Start the server
uvicorn server:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs

# Optional: Streamlit comparison UI
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

---

## Monitoring

Prometheus scrapes `/metrics` every 5 seconds. Grafana auto-loads the **FastAPI LLM Monitoring** dashboard on startup.

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total requests by endpoint and HTTP status |
| `http_request_duration_seconds` | Histogram | Request latency per endpoint |
| `inference_tokens_generated` | Histogram | Tokens returned per inference call |
| `inference_latency_ms` | Histogram | End-to-end model call latency |
| `active_requests` | Gauge | In-flight requests |
| `model_load_status` | Gauge | 1 = ready, 0 = not ready |

---

## Links

- **Model on Hugging Face:** https://huggingface.co/harsh2five/mistral-fastapi-finetuned
- **W&B training run:** https://wandb.ai/meenaharsh5432-nit-kurukshetra/fastapi-llm-finetune
- **GitHub:** https://github.com/meenaharsh5432-tech/fastapi-llm-mlops

---

## Resume Bullets

- Fine-tuned Mistral 7B on 32,575 FastAPI-specific QA pairs using QLoRA 4-bit quantisation (PEFT/LoRA), achieving eval loss of 0.97 and ~2s inference latency via Groq
- Built a production FastAPI inference server with OpenAI-compatible `/v1/chat/completions` endpoint, full Pydantic validation, and async request handling
- Instrumented the API with six Prometheus metrics (counters, histograms, gauges) and a pre-built Grafana dashboard covering request rate, P99 latency, and token throughput
- Containerised the full stack (FastAPI + Prometheus + Grafana) with Docker Compose and added a four-job GitHub Actions CI pipeline (test, lint, docker-build, notify)
- Curated and cleaned a 32K+ FastAPI QA dataset from official docs, GitHub issues, and Stack Overflow; published adapter weights and dataset to Hugging Face Hub


## Prometheus DashBoard
<img width="1918" height="912" alt="prometheus" src="https://github.com/user-attachments/assets/d471afe6-c140-4ba3-9ca0-35cf261be94d" />

## Grafana DashBoard
<img width="1543" height="737" alt="grafana_dashboard" src="https://github.com/user-attachments/assets/ebbfa169-c83c-4753-b9c3-191669064995" />
