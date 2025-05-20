import logging
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def summarize_paragraph(paragraph, sentences_count=2):
    parser = PlaintextParser.from_string(paragraph, Tokenizer("english"))

    summarizer = LsaSummarizer()

    summary = summarizer(parser.document, sentences_count)
    return summary

def generate_lsa_summary(text: str, num_sentences: int = 3) -> str:
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
        summary_tup = summarize_paragraph(text, num_sentences)
        summary = ''
        for sentence in summary_tup:
            summary += str(sentence) + '\n'

        return summary
    except Exception as e:
        logging.error(f"LSA summarization failed: {e}")
        return ""
