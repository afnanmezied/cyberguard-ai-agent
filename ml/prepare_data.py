
import pandas as pd

from paths import PHIUSIIL_RAW, PHIUSIIL_CLEAN


def clean_dataset(df):
    """Clean PhiUSIIL data for binary phishing detection."""

    # Keep only required columns
    df = df[["URL", "label"]].copy()

    # Remove surrounding whitespace
    df["URL"] = df["URL"].str.strip()

    # Check conflicting labels before deduplication
    conflicts = df.groupby("URL")["label"].nunique()

    if (conflicts > 1).any():
        raise ValueError("Conflicting labels found for the same URL")

    # Remove duplicate URLs
    df = df.drop_duplicates(subset="URL").copy()

    # Convert labels to CyberGuard convention
    # 0 = Benign, 1 = Phishing
    df["label"] = df["label"].map({1: 0, 0: 1})

    # Rename URL column
    df = df.rename(columns={"URL": "url"})

    return df


def validate_cleaned_data(df):
    """Check the cleaned dataset before saving."""

    if df.isna().any().any():
        raise ValueError("Missing values found")

    if df["url"].duplicated().any():
        raise ValueError("Duplicate URLs found")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Unexpected label values")


def main():
    # Load original dataset
    df = pd.read_csv(PHIUSIIL_RAW, keep_default_na=False)

    print("Original shape:", df.shape)

    # Clean and validate
    cleaned_df = clean_dataset(df)
    validate_cleaned_data(cleaned_df)

    # Save cleaned dataset
    PHIUSIIL_CLEAN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned_df.to_csv(PHIUSIIL_CLEAN, index=False)

    # Display summary
    print("\nCleaned shape:", cleaned_df.shape)
    print("\nClass distribution:")
    print(cleaned_df["label"].value_counts())
    print("\nDuplicate URLs:", cleaned_df["url"].duplicated().sum())
    print("Missing values:", cleaned_df.isna().sum().sum())


if __name__ == "__main__":
    main()
