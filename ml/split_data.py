
import pandas as pd
import tldextract

from sklearn.model_selection import GroupShuffleSplit
from paths import PHIUSIIL_FEATURES, PROCESSED_DIR, TRAIN_DATA, VALIDATION_DATA, TEST_DATA


extractor = tldextract.TLDExtract(suffix_list_urls=())
RANDOM_STATE = 42


def get_domain(url):
    """Extract the registered domain for grouping."""
    result = extractor(url)

    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}".lower()

    if result.ipv4 or result.ipv6:
        return str(result.ipv4 or result.ipv6)

    raise ValueError(f"Cannot extract domain: {url}")


def split_dataset(df):
    """Split the dataset without domain overlap."""
    groups = df["url"].apply(get_domain)

    # Split into training (70%) and temporary (30%)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    temp_groups = groups.iloc[temp_idx]

    # Split temporary data into validation and testing
    second_splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(second_splitter.split(temp_df, groups=temp_groups))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Verify domain separation
    train_domains = set(groups.iloc[train_idx])
    val_domains = set(temp_groups.iloc[val_idx])
    test_domains = set(temp_groups.iloc[test_idx])

    assert train_domains.isdisjoint(val_domains)
    assert train_domains.isdisjoint(test_domains)
    assert val_domains.isdisjoint(test_domains)

    return train_df, val_df, test_df


def main():
    df = pd.read_csv(PHIUSIIL_FEATURES, keep_default_na=False)
    train_df, val_df, test_df = split_dataset(df)

    datasets = {TRAIN_DATA: train_df, VALIDATION_DATA: val_df, TEST_DATA: test_df}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for output_path, subset in datasets.items():
        subset.to_csv(output_path, index=False)
        print(f"\n{output_path.stem.upper()}")
        print("Rows:", len(subset))
        print("Percentage:", round(len(subset) / len(df) * 100, 2))
        print("Labels:", subset["label"].value_counts().to_dict())

    print("\nDomain overlap: 0")
    print("Data splitting completed!")


if __name__ == "__main__":
    main()
