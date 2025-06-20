import logging
import json
import random
from math import floor
import pandas as pd
from tqdm import tqdm
from typing import Optional
from ollama_client import generate_llm_summary
from lsa_summarizer import generate_lsa_summary

import nltk
nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def clean_llm_output(text: str) -> str:
    """
    Remove filler phrases (e.g., "Sure! Here's...") from LLM-generated summaries.
    """
    lower_text = text.lower()
    if lower_text.startswith("sure") or lower_text.startswith("okay") or lower_text.startswith("here"):
        split_index = text.find("\n\n")
        if split_index != -1:
            text = text[split_index:].strip()

    search_strings = ["\n\nlet me know", "\n\nwould you like me to", "\n\n---\n\n", "\n\n**"]
    for search_string in search_strings:
        split_index = lower_text.find(search_string)
        if split_index != -1:
            return text[:split_index].strip()
    return text.strip()

def generate_fixed_random_list(seed, count, start, end):
    random.seed(seed)
    return [random.randint(start, end) for _ in range(count)]

def generate_and_save_summaries(
    input_csv: str,
    llm_json: str,
    lsa_json: str,
    text_column: str = "text",
    limit: Optional[int] = None,
    batch_no: Optional[int] = None
) -> None:
    """
    Generate LLM and BERT summaries from a CSV file and save them to JSON files.
    """
    logging.info(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)

    llm_summaries, lsa_summaries = {}, {}


    if batch_no > 4:
        logging.warning(f"[Batch no. failed] batch_no should be <= 4, but got {batch_no}.")
    else:
        start = (len(df)/5) *batch_no
        end = (len(df)/5) * (batch_no + 1)
        idx_list = generate_fixed_random_list(1276672,2000, int(start), floor(end))
        df = df.iloc[idx_list]
    logging.info("🧠 Generating summaries...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(idx)
        text = str(row.get(text_column, "")).strip()
        if not text or len(text.split()) < 10:
            continue  # skip short/empty inputs

        # LLM summary
        try:
            llm_raw = generate_llm_summary(text, doc_id=doc_id)
            llm_summary = clean_llm_output(llm_raw)
            if llm_summary:
                llm_summaries[doc_id] = llm_summary
        except Exception as e:
            logging.warning(f"[LLM failed] ID={doc_id} | {e}")

        # LSA summary
        try:
            lsa_summary = generate_lsa_summary(text)
            if lsa_summary and len(lsa_summary.strip()) > 10:
                lsa_summaries[doc_id] = lsa_summary.strip()
        except Exception as e:
            logging.warning(f"[LSA failed] ID={doc_id} | {e}")

        # Save JSON files
        with open(llm_json, 'w', encoding='utf-8') as f:
            json.dump(llm_summaries, f, ensure_ascii=False, indent=2)
        with open(lsa_json, 'w', encoding='utf-8') as f:
            json.dump(lsa_summaries, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Saved {len(llm_summaries)} LLM summaries → {llm_json}")
    logging.info(f"✅ Saved {len(lsa_summaries)} LSA summaries → {lsa_json}")


import concurrent.futures
from functools import partial

def generate_and_save_summaries_faster(
    input_csv: str,
    llm_json: str,
    lsa_json: str,
    text_column: str = "text",
    max_workers: int = 32
) -> None:
    """
    Generate LLM and BERT summaries from a CSV file using multithreading.
    """
    logging.info(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    llm_summaries, lsa_summaries = {}, {}

    # Worker function that processes a single row
    def process_row(row, idx):
        doc_id = str(idx)
        text = str(row.get(text_column, "")).strip()
        results = {}
        
        if not text or len(text.split()) < 10:
            return results  # skip short/empty inputs

        # LLM summary
        try:
            llm_raw = generate_llm_summary(text, doc_id=doc_id)
            llm_summary = clean_llm_output(llm_raw)
            if llm_summary:
                results['llm'] = (doc_id, llm_summary)
        except Exception as e:
            logging.warning(f"[LLM failed] ID={doc_id} | {e}")

        # LSA summary
        try:
            lsa_summary = generate_lsa_summary(text)
            if lsa_summary and len(lsa_summary.strip()) > 10:
                results['lsa'] = (doc_id, lsa_summary.strip())
        except Exception as e:
            logging.warning(f"[LSA failed] ID={doc_id} | {e}")
            
        return results

    # Process rows in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            futures.append(executor.submit(process_row, row, idx))
            
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results = future.result()
            if 'llm' in results:
                doc_id, summary = results['llm']
                llm_summaries[doc_id] = summary
            if 'lsa' in results:
                doc_id, summary = results['lsa']
                lsa_summaries[doc_id] = summary
            
            # Periodic saving (every 10 processed items)
            if len(llm_summaries) % 10 == 0 or len(lsa_summaries) % 10 == 0:
                with open(llm_json, 'w', encoding='utf-8') as f:
                    json.dump(llm_summaries, f, ensure_ascii=False, indent=2)
                with open(lsa_json, 'w', encoding='utf-8') as f:
                    json.dump(lsa_summaries, f, ensure_ascii=False, indent=2)

    # Final save
    with open(llm_json, 'w', encoding='utf-8') as f:
        json.dump(llm_summaries, f, ensure_ascii=False, indent=2)
    with open(lsa_json, 'w', encoding='utf-8') as f:
        json.dump(lsa_summaries, f, ensure_ascii=False, indent=2)

    # logging.info(f"✅ Saved {len(llm_summaries)} LLM summaries → {llm_json}")
    logging.info(f"✅ Saved {len(lsa_summaries)} LSA summaries → {lsa_json}")

if __name__ == "__main__":
    generate_and_save_summaries_faster(
        input_csv="data/cleaned_agnews_light.csv",
        llm_json=f"data/summaries_agnews_llm.json",
        lsa_json=f"data/summaries_agnews_lsa.json")
