import logging
import pandas as pd
from pathlib import Path
from typing import Union

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def check_dataset(path: Union[str, Path], text_col="text", label_col="label"):
    path = Path(path)
    logging.info(f"🔍 Checking: {path.name}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"❌ Could not read {path.name}: {e}")
        return

    logging.info(f"📦 Total rows: {len(df)}")

    # Missing & empty values
    missing_text = df[text_col].isna().sum()
    empty_text = (df[text_col].astype(str).str.strip() == "").sum()
    logging.warning(f"⚠️ Missing texts: {missing_text}")
    logging.warning(f"⚠️ Empty texts: {empty_text}")

    # Length stats
    df["length"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    logging.info("📏 Text Length Stats:\n" + str(df["length"].describe()))
    short_count = (df["length"] < 10).sum()
    logging.warning(f"⚠️ Very short (<10 words): {short_count}")

    # Label stats
    if label_col in df.columns:
        logging.info("🏷️ Label Distribution:\n" +
                     str(df[label_col].value_counts()))
    else:
        logging.warning("⚠️ No label column found")

    # Duplicates
    duplicates = df.duplicated(subset=[text_col]).sum()
    logging.warning(f"⚠️ Duplicate texts: {duplicates}")

    # Sample preview
    logging.info("🧪 Sample rows:\n" +
                 str(df[[text_col, label_col]].sample(3, random_state=42)))


if __name__ == "__main__":
    paths = [
        "data/cleaned_20news.csv",
        "data/cleaned_agnews.csv",
    ]
    for path in paths:
        check_dataset(path)
