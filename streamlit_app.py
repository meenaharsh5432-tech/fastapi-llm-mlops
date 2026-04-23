import streamlit as st
import requests
import os

# ─── Config ──────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a FastAPI expert assistant fine-tuned on 
32,000+ FastAPI-specific QA pairs from official docs, GitHub issues, 
and Stack Overflow. Give specific, accurate, version-aware FastAPI 
answers with complete working code examples. Always include runnable code."""

# Page config 
st.set_page_config(
    page_title="FastAPI Domain LLM",
    page_icon="🚀",
    layout="wide"
)

# Sidebar 
with st.sidebar:
    st.title("FastAPI Domain LLM")
    st.markdown("""
    **Mistral 7B fine-tuned on 32,000+ FastAPI QA pairs**
    
    Training data:
    - FastAPI official docs
    - GitHub issues + PRs
    - Stack Overflow answers
    - HuggingFace datasets
    
    ---
    
    **Links:**
    - [Model on HuggingFace](https://huggingface.co/harsh2five/mistral-fastapi-finetuned)
    - [Training on W&B](https://wandb.ai/meenaharsh5432-nit-kurukshetra/fastapi-llm-finetune)
    - [GitHub](https://github.com/meenaharsh5432-tech/fastapi-llm-mlops)
    
    ---
    
    **Training details:**
    - Base: Mistral 7B Instruct v0.3
    - Method: QLoRA (4-bit)
    - LoRA rank: 16
    - Eval loss: 0.97
    - Dataset: 32,575 pairs
    """)
    
    st.divider()
    max_tokens = st.slider("Max tokens", 100, 1024, 512)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)

# Main UI 
st.title("FastAPI Domain LLM — Ask Anything")
st.caption("Fine-tuned Mistral 7B on 32,000+ FastAPI-specific QA pairs")

# Example questions
examples = [
    "How do I add CORS middleware to FastAPI?",
    "How do I implement JWT authentication in FastAPI?",
    "How do I create a POST endpoint that accepts a JSON body?",
    "What is dependency injection in FastAPI?",
    "How do I add rate limiting to FastAPI?",
    "How do I handle file uploads in FastAPI?",
    "How do I connect FastAPI to PostgreSQL?",
    "How do I use background tasks in FastAPI?",
    "How do I implement WebSockets in FastAPI?",
    "How do I add request validation in FastAPI?",
]

st.markdown("**Example questions:**")
cols = st.columns(5)
for i, ex in enumerate(examples[:5]):
    with cols[i]:
        if st.button(ex, use_container_width=True, key=f"ex_{i}"):
            st.session_state.question = ex

cols2 = st.columns(5)
for i, ex in enumerate(examples[5:]):
    with cols2[i]:
        if st.button(ex, use_container_width=True, key=f"ex2_{i}"):
            st.session_state.question = ex

st.divider()

# Question input
question = st.text_area(
    "Ask a FastAPI question:",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. How do I implement OAuth2 with refresh tokens in FastAPI?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask", type="primary", use_container_width=True)

# Generate response 
def generate(question, max_tokens, temperature):
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not configured."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature)
    }
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# Response 
if ask and question.strip():
    with st.spinner("Generating answer..."):
        answer = generate(question, max_tokens, temperature)
    
    st.markdown("### Answer")
    st.markdown(answer)
    
    # Save to history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "question": question,
        "answer": answer
    })

# History 
if st.session_state.get("history"):
    st.divider()
    st.subheader("Previous Questions")
    for item in reversed(st.session_state.history[-5:]):
        with st.expander(item["question"][:80] + "..."):
            st.markdown(item["answer"])