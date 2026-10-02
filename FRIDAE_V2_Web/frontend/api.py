import os
import requests

API_URL = os.getenv("FRIDAE_API_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("FRIDAE_API_TOKEN", "")

def _request(method, path, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    if API_TOKEN:
        headers["x-eval-token"] = API_TOKEN
    timeout = kwargs.pop("timeout", 120)
    response = requests.request(
        method, f"{API_URL}{path}",
        headers=headers, timeout=timeout, **kwargs
    )
    response.raise_for_status()
    return response

def health():
    return _request("GET", "/health", timeout=20).json()

def chat(question, chat_id):
    return _request("POST", "/api/chat",
                    json={"question": question, "chat_id": chat_id}).json()

def upload(file_name, data, content_type, chat_id):
    return _request("POST", "/api/upload", params={"chat_id": chat_id},
                    files={"file": (file_name, data, content_type)}).json()

def dataset(chat_id):
    return _request("GET", "/api/dataset", params={"chat_id": chat_id},
                    timeout=20).json()

def artifacts(chat_id):
    return _request("GET", "/api/artifacts", params={"chat_id": chat_id},
                    timeout=20).json()

def artifact_bytes(chat_id, artifact_id):
    return _request("GET", f"/api/artifacts/{artifact_id}",
                    params={"chat_id": chat_id}, timeout=60).content

def logs():
    return _request("GET", "/run.jsonl", timeout=20).text
