from llm.summarize_and_vectorize import summarize_and_vectorize


def test_summarize_and_vectorize():
    text = (
        "Artificial intelligence is revolutionizing education, healthcare, and business. "
        "Large language models like ChatGPT are being used for writing, tutoring, coding, and analysis."
    )

    result = summarize_and_vectorize(text)

    assert isinstance(result["bert_summary"], str) and len(
        result["bert_summary"]) > 0
    assert isinstance(result["llm_summary"], str) and len(
        result["llm_summary"]) > 0
    assert isinstance(result["bert_vector"], list) and len(
        result["bert_vector"]) > 0
    assert isinstance(result["llm_vector"], list) and len(
        result["llm_vector"]) > 0
