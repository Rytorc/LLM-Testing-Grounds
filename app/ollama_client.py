import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate(model, prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

def generate_stream(model, prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
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

        data = json.loads(line)

        if "response" in data:
            token = data ["response"]

            print(token, end="", flush=True)

            full_response += token

    print()

    return full_response