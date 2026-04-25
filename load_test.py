
import requests
import time

SERVER = "http://localhost:8000"

questions = [
    "How do I add CORS to FastAPI?",
    "How do I implement JWT authentication in FastAPI?",
    "How do I create a POST endpoint with JSON body?",
    "What is dependency injection in FastAPI?",
    "How do I add rate limiting to FastAPI?",
    "How do I handle file uploads in FastAPI?",
    "How do I connect FastAPI to PostgreSQL?",
    "How do I use background tasks in FastAPI?",
    "How do I implement WebSockets in FastAPI?",
    "How do I add request validation in FastAPI?",
]

print("Sending test requests...")
for i, q in enumerate(questions * 2):  # 20 requests
    try:
        r = requests.post(f"{SERVER}/chat", 
                         json={"message": q, "max_tokens": 200},
                         timeout=30)
        print(f"Request {i+1}/20: {r.status_code} — {r.json()['latency_ms']:.0f}ms")
        time.sleep(2)
    except Exception as e:
        print(f"Error: {e}")

print("Done — check Grafana now!")