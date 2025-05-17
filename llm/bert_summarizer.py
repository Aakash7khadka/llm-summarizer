import logging
from summarizer import Summarizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

# Load BERT summarizer once
bert_model = Summarizer()


def generate_bert_summary(text: str, num_sentences: int = 3) -> str:
    """
    Generate an extractive summary using a BERT-based model.

    Args:
        text (str): Input document text.
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: Extractive summary, or an empty string on failure.
    """
    if not text or len(text.split()) < 10:
        logging.warning("Text too short for meaningful summarization.")
        return ""

    try:
        summary = bert_model(text, num_sentences=num_sentences)
        return summary.strip()
    except Exception as e:
        logging.error(f"BERT summarization failed: {e}")
        return ""
