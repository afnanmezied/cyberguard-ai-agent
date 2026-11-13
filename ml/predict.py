
import argparse
import joblib
import pandas as pd

from paths import RANDOM_FOREST_MODEL
from feature_extraction import extract_url_features


def load_model():
    """Load the trained Random Forest model."""
    if not RANDOM_FOREST_MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {RANDOM_FOREST_MODEL}")

    return joblib.load(RANDOM_FOREST_MODEL)


def predict_url(url, model):
    """Predict whether a URL is benign or phishing."""
    features = extract_url_features(url)
    data = pd.DataFrame([features])

    prediction = int(model.predict(data)[0])
    phishing_score = float(model.predict_proba(data)[0, 1])

    return {
        "url": url.strip(),
        "prediction": prediction,
        "label": "Phishing" if prediction == 1 else "Benign",
        "phishing_score": round(phishing_score, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="CyberGuard URL Phishing Detection")
    parser.add_argument("url", help="URL to analyze")
    args = parser.parse_args()

    model = load_model()
    result = predict_url(args.url, model)

    print("\n========== CYBERGUARD PREDICTION ==========")
    print("URL:", result["url"])
    print("Prediction:", result["label"])
    print("Phishing Score:", f"{result['phishing_score']:.2%}")


if __name__ == "__main__":
    main()
