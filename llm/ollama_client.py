import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def generate_llm_summary(text: str, model: str = "gemma:2b", n_sentences: int = 4) -> str:
    """
    Generate an abstractive summary using a local LLM served via Ollama.

    Args:
        text (str): The input document to summarize.
        model (str): The Ollama model name to use (e.g., "gemma:2b").
        n_sentences (int): Desired number of output sentences.

    Returns:
        str: The generated summary, or an empty string on failure.
    """
    if not text or len(text.split()) < 10:
        logging.warning("Text too short for LLM summarization.")
        return ""

    prompt = f"Summarize the following text in {n_sentences} short sentences:\n\n{text}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=15
        )

        if response.ok:
            summary = response.json().get("response", "").strip()
            return summary
        else:
            logging.warning(
                f"Ollama API returned status {response.status_code}: {response.text}")
            return ""

    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama API request failed: {e}")
        return ""
