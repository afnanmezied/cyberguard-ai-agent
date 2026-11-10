
import pandas as pd
from paths import PHIUSIIL_RAW, PHIUSIIL_CLEAN


def explore_data(df):
    """Explore the original PhiUSIIL dataset."""
    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 Rows:")
    print(
        df[["URL", "Domain", "TLD", "URLLength", "label"]]
        .head()
        .to_string(index=False)
    )


def check_data_quality(df):
    """Check missing values, duplicates and labels."""
    print("\n========== DATA QUALITY ==========")
    print("\nMissing Values:")
    print(df.isna().sum().to_string())

    print("\nDuplicate Rows:", df.duplicated().sum())
    print("Duplicate URLs:", df["URL"].duplicated().sum())

    print("\nClass Distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nClass Percentages:")
    print(df["label"].value_counts(normalize=True) * 100)

    conflicts = df.groupby("URL")["label"].nunique()
    print("\nConflicting URL Labels:", (conflicts > 1).sum())

    duplicate_urls = df[df.duplicated("URL", keep=False)]
    print("\nDuplicate URL Labels:")
    print(duplicate_urls["label"].value_counts())


def clean_data(df):
    """Clean the dataset and standardize labels."""
    df = df[["URL", "label"]].copy()

    df["URL"] = df["URL"].str.strip()

    if df["URL"].eq("").any():
        raise ValueError("Empty URLs found")

    conflicts = df.groupby("URL")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting labels found")

    df = df.drop_duplicates(subset="URL").copy()

    # PhiUSIIL: 1 = Legitimate, 0 = Phishing
    # CyberGuard: 0 = Benign, 1 = Phishing
    df["label"] = df["label"].map({1: 0, 0: 1})

    df = df.rename(columns={"URL": "url"})

    return df


def validate_cleaned_data(df):
    """Validate the cleaned data before saving."""
    if df.isna().any().any():
        raise ValueError("Missing values found")

    if df["url"].duplicated().any():
        raise ValueError("Duplicate URLs found")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Unexpected labels found")

    print("\n========== CLEANED DATA ==========")
    print("Shape:", df.shape)
    print("\nClass Distribution:")
    print(df["label"].value_counts())
    print("Duplicate URLs:", df["url"].duplicated().sum())
    print("Missing Values:", df.isna().sum().sum())


def main():
    df = pd.read_csv(PHIUSIIL_RAW, keep_default_na=False)

    explore_data(df)
    check_data_quality(df)

    cleaned_df = clean_data(df)
    validate_cleaned_data(cleaned_df)

    PHIUSIIL_CLEAN.parent.mkdir(
        parents=True, exist_ok=True
    )
    cleaned_df.to_csv(PHIUSIIL_CLEAN, index=False)

    print("\nSaved to:", PHIUSIIL_CLEAN)


if __name__ == "__main__":
    main()
