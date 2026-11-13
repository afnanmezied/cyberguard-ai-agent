
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"

SYSTEM_PROMPT = """
You are CyberGuard, a helpful cybersecurity assistant.
Answer cybersecurity questions clearly and concisely.
Do not claim to have scanned a website or verified its safety.
If discussing a URL, explain that URL-based predictions
are estimates and not guarantees.
"""


def chat_with_llm(message):
    """Send a message to the local Qwen model."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message.strip()}
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.RequestException as error:
        raise RuntimeError(f"Ollama connection failed: {error}") from error


def main():
    question = input("Ask CyberGuard: ")
    answer = chat_with_llm(question)
    print("\nCyberGuard:", answer)


if __name__ == "__main__":
    main()
