
import re

from predict import load_model, predict_url
from llm import chat_with_llm


def extract_url(message):
    """Find an HTTP or HTTPS URL in a message."""
    match = re.search(r"https?://[^\s<>\"']+", message, re.IGNORECASE)
    return match.group(0).rstrip(".,;!?)]}") if match else None


def run_agent(message, model):
    """Route user messages to the appropriate tool."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty")

    url = extract_url(message)

    if not url:
        return chat_with_llm(message)

    result = predict_url(url, model)

    explanation_prompt = f"""
Explain this URL classification result briefly.

URL: {result['url']}
Model prediction: {result['label']}
Phishing score: {result['phishing_score']:.2%}

The prediction comes from a URL-based Random Forest classifier.
Do not change the classification or invent evidence.
Do not claim the website was visited or scanned.
Explain that the prediction is not a safety guarantee.
Provide practical safety advice in 2-3 sentences.
"""

    try:
        explanation = chat_with_llm(explanation_prompt)
    except RuntimeError:
        explanation = "Explanation unavailable. Avoid entering sensitive information into unverified websites."

    return (
        f"URL: {result['url']}\n"
        f"Prediction: {result['label']}\n"
        f"Phishing Score: {result['phishing_score']:.2%}\n\n"
        f"Explanation: {explanation}"
    )


def main():
    model = load_model()
    print("CyberGuard AI Agent is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        message = input("You: ").strip()

        if message.lower() in ("exit", "quit"):
            break

        try:
            answer = run_agent(message, model)
            print(f"\nCyberGuard: {answer}\n")
        except (ValueError, RuntimeError) as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
