import logging
import json
import pandas as pd
from tqdm import tqdm
from typing import Optional
from llm.ollama_client import generate_llm_summary
from llm.bert_summarizer import generate_bert_summary

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def clean_llm_output(text: str) -> str:
    """
    Remove filler phrases (e.g., "Sure! Here's...") from LLM-generated summaries.
    """
    if text.lower().startswith("sure"):
        split_index = text.find("\n\n")
        if split_index != -1:
            return text[split_index:].strip()
    return text.strip()


def generate_and_save_summaries(
    input_csv: str,
    llm_json: str,
    bert_json: str,
    text_column: str = "text",
    limit: Optional[int] = None
) -> None:
    """
    Generate LLM and BERT summaries from a CSV file and save them to JSON files.
    """
    logging.info(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)

    llm_summaries, bert_summaries = {}, {}

    logging.info("🧠 Generating summaries...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(idx)
        text = str(row.get(text_column, "")).strip()

        if not text or len(text.split()) < 10:
            continue  # skip short/empty inputs

        # LLM summary
        try:
            llm_raw = generate_llm_summary(text)
            llm_summary = clean_llm_output(llm_raw)
            if llm_summary:
                llm_summaries[doc_id] = llm_summary
        except Exception as e:
            logging.warning(f"[LLM failed] ID={doc_id} | {e}")

        # BERT summary
        try:
            bert_summary = generate_bert_summary(text)
            if bert_summary and len(bert_summary.strip()) > 10:
                bert_summaries[doc_id] = bert_summary.strip()
        except Exception as e:
            logging.warning(f"[BERT failed] ID={doc_id} | {e}")

    # Save JSON files
    with open(llm_json, 'w', encoding='utf-8') as f:
        json.dump(llm_summaries, f, ensure_ascii=False, indent=2)
    with open(bert_json, 'w', encoding='utf-8') as f:
        json.dump(bert_summaries, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Saved {len(llm_summaries)} LLM summaries → {llm_json}")
    logging.info(f"✅ Saved {len(bert_summaries)} BERT summaries → {bert_json}")


if __name__ == "__main__":
    generate_and_save_summaries(
        input_csv="data/cleaned_20news.csv",
        llm_json="data/summaries_llm.json",
        bert_json="data/summaries_bert.json",
        limit=200  # Set to None to process all
    )
