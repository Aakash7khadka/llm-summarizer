import logging
from llm.bert_summarizer import generate_bert_summary

logging.basicConfig(level=logging.INFO)


def test_bert_summary():
    text = """
    The stock market rallied today as investors responded positively to new inflation data
    indicating a slowdown in consumer prices. Major indexes rose more than 2%,
    marking the best day in weeks for Wall Street.
    """
    summary = generate_bert_summary(text, num_sentences=3)
    logging.info(f"Generated summary:\n{summary}")


if __name__ == "__main__":
    test_bert_summary()
