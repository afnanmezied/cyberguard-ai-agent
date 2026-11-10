
import numpy as np
import pandas as pd

from paths import PHIUSIIL_CLEAN, PHIUSIIL_FEATURES
from feature_extraction import extract_url_features


NUMERIC_FEATURES = [
    "url_length",
    "num_dots",
    "has_https",
    "has_ip",
    "num_subdirs",
    "num_params",
    "suspicious_words",
    "special_char_count",
    "digits_count",
    "entropy",
]


def build_feature_dataset(df):
    """Extract URL features from the cleaned dataset."""
    records = []
    invalid_count = 0

    for url, label in df[["url", "label"]].itertuples(
        index=False, name=None
    ):
        try:
            url = url.strip()
            features = extract_url_features(url)

            features["url"] = url
            features["label"] = label

            records.append(features)

        except (ValueError, TypeError) as error:
            invalid_count += 1
            print(f"Skipped invalid URL: {error}")

    if not records:
        raise ValueError("No valid URL records found")

    result = pd.DataFrame(records)

    print("Invalid URLs:", invalid_count)

    return result


def validate_feature_dataset(df):
    """Validate the extracted features before saving."""
    print("\n========== FEATURE VALIDATION ==========")

    # Check expected columns
    required_columns = set(
        NUMERIC_FEATURES + ["tld", "url", "label"]
    )

    if set(df.columns) != required_columns:
        raise ValueError("Unexpected feature columns")

    # Check missing and empty values
    if df.isna().any().any():
        raise ValueError("Missing values found")

    if df["url"].eq("").any() or df["tld"].eq("").any():
        raise ValueError("Empty URL or TLD found")

    # Check duplicate URLs
    if df["url"].duplicated().any():
        raise ValueError("Duplicate URLs found")

    # Check labels
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Invalid label values")

    # Check numerical values
    if not np.isfinite(
        df[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite numeric values found")

    if (df[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError("Negative feature values found")

    # Check binary features
    for column in ["has_https", "has_ip"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"Invalid values in {column}")

    # Check URL length consistency
    if not (
        df["url_length"] == df["url"].str.len()
    ).all():
        raise ValueError("URL length mismatch found")

    print("All validation checks passed!")


def display_summary(df):
    """Display the processed dataset summary."""
    print("\n========== FEATURE DATASET SUMMARY ==========")

    print("Shape:", df.shape)

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    print("\nMissing Values:", df.isna().sum().sum())
    print("Duplicate URLs:", df["url"].duplicated().sum())

    print("\nNumeric Feature Statistics:")
    print(df[NUMERIC_FEATURES].describe().T)

    print("\nHTTPS Distribution:")
    print(df["has_https"].value_counts())

    print("\nTop 10 TLDs:")
    print(df["tld"].value_counts().head(10))


def main():
    # Load cleaned data
    df = pd.read_csv(
        PHIUSIIL_CLEAN,
        keep_default_na=False
    )

    print("Cleaned Dataset Shape:", df.shape)

    # Build feature dataset
    feature_df = build_feature_dataset(df)

    # Validate before saving
    validate_feature_dataset(feature_df)

    # Display summary
    display_summary(feature_df)

    # Save only after successful validation
    PHIUSIIL_FEATURES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        PHIUSIIL_FEATURES,
        index=False
    )

    print("\nSaved to:", PHIUSIIL_FEATURES)


if __name__ == "__main__":
    main()
