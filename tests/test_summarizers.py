from llm.lsa_summarizer import generate_lsa_summary
from llm.ollama_client import generate_llm_summary

sample_text = (
    "Artificial intelligence is transforming every industry including "
    "healthcare, finance, and education. "
    "It is enabling new levels of automation, prediction, and personalization."
)


def test_bert_summary():
    summary = generate_lsa_summary(sample_text)
    assert isinstance(summary, str)
    assert len(summary.split()) > 3


def test_llm_summary():
    summary = generate_llm_summary(sample_text)
    assert isinstance(summary, str)
    assert len(summary.split()) > 3
