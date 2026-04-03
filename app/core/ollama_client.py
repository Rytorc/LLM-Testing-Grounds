import requests
import json

from app.config import settings

OLLAMA_URL = settings.ollama_url
REQUEST_TIMEOUT = settings.request_timeout

class OllamaClientError(Exception):
    pass

def _post(payload, stream=False):
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=stream,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        raise OllamaClientError(f"Ollama request failed: {e}") from e

def generate(model, prompt):
    response = _post(
        {
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        stream=False,
    )

    data = response.json()
    return data.get("response", "").strip()

def generate_stream(model, prompt):
    response = _post(
        {
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    full_response = ""

    for line in response.iter_lines():
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        token = data.get("response")
        if token:
            full_response += token

    return full_response.strip()