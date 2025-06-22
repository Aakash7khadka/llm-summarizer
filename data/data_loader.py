import logging
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
from data.text_cleaner import clean_text_light, clean_text_strict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def filter_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() > 0]
    df = df[df[text_column].str.split().str.len() >= 10]
    df = df.drop_duplicates(subset=[text_column])
    return df


def load_and_clean_20newsgroups(save_path='data/cleaned_20news.csv', cleaning_mode='strict', light_version = False):
    logging.info("Loading 20 Newsgroups dataset...")
    raw = fetch_20newsgroups(subset='all', remove=(
        'headers', 'footers', 'quotes'))
    cleaner = clean_text_strict if cleaning_mode == 'strict' else clean_text_light
    df = pd.DataFrame({
        'text': [cleaner(text) for text in raw.data],
        'label': raw.target,
        'label_name': [raw.target_names[i] for i in raw.target]
    })
    df = filter_dataframe(df)

    if light_version: df = df.head(2300)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved cleaned 20 Newsgroups to: {save_path}")


def load_and_clean_agnews(save_path='data/cleaned_agnews.csv', cleaning_mode='strict', light_version = False):
    logging.info("Loading AG News dataset...")
    dataset = load_dataset('ag_news', split='train')
    cleaner = clean_text_strict if cleaning_mode == 'strict' else clean_text_light
    texts = [cleaner(example['text']) for example in dataset]
    labels = [example['label'] for example in dataset]
    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'label_name': [label_names[i] for i in labels]
    })
    df = filter_dataframe(df)

    if light_version: df = df.head(2300)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved cleaned AG News to: {save_path}")


if __name__ == "__main__":
    load_and_clean_20newsgroups('data/cleaned_20news_strict.csv', 'strict')
    load_and_clean_agnews('data/cleaned_agnews_strict.csv', 'strict')
    load_and_clean_20newsgroups('data/cleaned_20news_light.csv', 'light')
    load_and_clean_agnews('data/cleaned_agnews_light.csv', 'light')
