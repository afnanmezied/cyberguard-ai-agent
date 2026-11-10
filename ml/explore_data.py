import pandas as pd
from paths import PHIUSIIL_RAW


def explore_dataset(df):
    """Display dataset structure and basic information."""
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
    """Check missing values and duplicate records."""
    print("\n========== DATA QUALITY ==========")
    print("\nMissing Values:")
    print(df.isna().sum().to_string())

    print("\nDuplicate Rows:", df.duplicated().sum())
    print("Duplicate URLs:", df["URL"].duplicated().sum())


def analyze_labels(df):
    """Analyze class distribution and conflicting labels."""
    print("\n========== LABEL ANALYSIS ==========")

    print("\nClass Distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nClass Percentages:")
    print(df["label"].value_counts(normalize=True) * 100)

    conflicts = df.groupby("URL")["label"].nunique()
    print("\nConflicting URL Labels:", (conflicts > 1).sum())

    duplicate_urls = df[df.duplicated("URL", keep=False)]

    print("\nDuplicate URL Labels:")
    print(duplicate_urls["label"].value_counts())


def main():
    df = pd.read_csv(PHIUSIIL_RAW, keep_default_na=False)

    explore_dataset(df)
    check_data_quality(df)
    analyze_labels(df)


if __name__ == "__main__":
    main()
