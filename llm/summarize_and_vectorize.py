from llm.bert_summarizer import generate_bert_summary
from llm.ollama_client import generate_llm_summary
from sentence_transformers import SentenceTransformer

# Load BERT embedder once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def summarize_and_vectorize(text: str) -> dict:
    """
    Generate summaries (LLM + BERT) and their vector representations.

    Args:
        text (str): Raw document text

    Returns:
        dict with keys:
            - 'bert_summary': str
            - 'llm_summary': str
            - 'bert_vector': List[float]
            - 'llm_vector': List[float]
    """
    if not text or len(text.split()) < 10:
        return {
            "bert_summary": "",
            "llm_summary": "",
            "bert_vector": [],
            "llm_vector": [],
            "error": "Text too short for summarization."
        }

    # Generate summaries
    bert_summary = generate_bert_summary(text)
    llm_summary = generate_llm_summary(text)

    # Embed summaries
    bert_vector = embedding_model.encode(
        bert_summary).tolist() if bert_summary else []
    llm_vector = embedding_model.encode(
        llm_summary).tolist() if llm_summary else []

    return {
        "bert_summary": bert_summary,
        "llm_summary": llm_summary,
        "bert_vector": bert_vector,
        "llm_vector": llm_vector
    }
