"""
Run: streamlit run app.py
"""

import streamlit as st
import requests
import time

# ─── Page config 
st.set_page_config(
    page_title="FastAPI LLM — Base vs Fine-tuned",
    page_icon="🚀",
    layout="wide"
)

st.title("FastAPI Domain LLM — Model Comparison")
st.caption("Base Mistral 7B vs Fine-tuned on FastAPI/Python domain data")

# ─── Sidebar 
with st.sidebar:
    st.header("Settings")
    server_url = st.text_input("Server URL", value="http://localhost:8000")
    max_tokens = st.slider("Max tokens", 100, 1024, 512)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)

    st.divider()
    st.header("Groq API Key")
    groq_api_key = st.text_input(
        "API Key (for base model)",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at groq.com to enable side-by-side comparison"
    )

    st.divider()
    st.header("Example Questions")
    examples = [
        "How do I add CORS to FastAPI?",
        "How do I create a POST endpoint with a JSON body?",
        "What is dependency injection in FastAPI?",
        "How do I add JWT authentication to FastAPI?",
        "How do I handle file uploads in FastAPI?",
        "How do I use background tasks in FastAPI?",
        "What is the difference between async and sync endpoints?",
        "How do I add request validation in FastAPI?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex

# ─── Main input ──────────────────────────────────────────
question = st.text_area(
    "Ask a FastAPI question:",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. How do I add authentication to FastAPI?"
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    ask_button = st.button("Ask Model", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.question = ""
    st.rerun()

# ─── Response 
if ask_button and question:
    # Check server health
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        server_ok = health.json().get("model_loaded", False)
    except:
        server_ok = False

    if not server_ok:
        st.error("Server not running. Start it with: uvicorn server:app --port 8000")
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("Your Fine-tuned Model")
            st.caption("Mistral 7B + LoRA adapter trained on FastAPI data")
            with st.spinner("Generating..."):
                start = time.time()
                try:
                    r = requests.post(
                        f"{server_url}/chat",
                        json={
                            "message": question,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        },
                        timeout=120
                    )
                    elapsed = time.time() - start

                    if r.status_code != 200:
                        detail = r.json().get("detail", r.text)
                        st.error(f"Server error {r.status_code}: {detail}")
                    else:
                        data = r.json()
                        st.success("Response:")
                        st.markdown(data["response"])
                        st.caption(f"Tokens: {data['tokens_generated']} | "
                                  f"Latency: {data['latency_ms']:.0f}ms")
                except Exception as e:
                    st.error(f"Error: {e}")

        with right:
            st.subheader("Base Model (Groq API)")
            st.caption("Untuned LLaMA 3.3 70B for comparison")
            if not groq_api_key:
                st.info("Enter your Groq API key in the sidebar to enable base model comparison")
            else:
                with st.spinner("Generating..."):
                    start = time.time()
                    try:
                        r = requests.post(
                            f"{server_url}/chat/base",
                            json={
                                "message": question,
                                "max_tokens": max_tokens,
                                "temperature": temperature
                            },
                            headers={"X-Groq-API-Key": groq_api_key},
                            timeout=120
                        )
                        elapsed = time.time() - start

                        if r.status_code != 200:
                            detail = r.json().get("detail", r.text)
                            st.error(f"Server error {r.status_code}: {detail}")
                        else:
                            data = r.json()
                            st.success("Response:")
                            st.markdown(data["response"])
                            st.caption(f"Tokens: {data['tokens_generated']} | "
                                      f"Latency: {data['latency_ms']:.0f}ms")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ─── History 
if "history" not in st.session_state:
    st.session_state.history = []

if ask_button and question and server_ok:
    st.session_state.history.append(question)

if st.session_state.get("history"):
    st.divider()
    st.subheader("Previous Questions")
    for q in reversed(st.session_state.history[-5:]):
        st.text(f"• {q}")
