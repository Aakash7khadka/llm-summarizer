import os
import logging
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
from data.text_cleaner import clean_text

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def filter_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Remove missing values, short texts, and duplicates.
    """
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() > 0]
    df = df[df[text_column].str.split().str.len() >= 10]
    df = df.drop_duplicates(subset=[text_column])
    return df


def load_and_clean_20newsgroups(save_path: str = 'data/cleaned_20news.csv') -> None:
    """
    Fetch, clean, and save the 20 Newsgroups dataset.
    """
    logging.info("📥 Loading 20 Newsgroups dataset...")
    raw = fetch_20newsgroups(subset='all', remove=(
        'headers', 'footers', 'quotes'))

    df = pd.DataFrame({
        'text': [clean_text(text) for text in raw.data],
        'label': raw.target,
        'label_name': [raw.target_names[i] for i in raw.target]
    })

    df = filter_dataframe(df)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logging.info(f"✅ Saved cleaned 20 Newsgroups to: {save_path}")


def load_and_clean_agnews(save_path: str = 'data/cleaned_agnews.csv') -> None:
    """
    Fetch, clean, and save the AG News dataset.
    """
    logging.info("📥 Loading AG News dataset...")
    dataset = load_dataset('ag_news', split='train')

    texts = [clean_text(example['text']) for example in dataset]
    labels = [example['label'] for example in dataset]
    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'label_name': [label_names[i] for i in labels]
    })

    df = filter_dataframe(df)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logging.info(f"✅ Saved cleaned AG News to: {save_path}")


if __name__ == "__main__":
    load_and_clean_20newsgroups()
    load_and_clean_agnews()
