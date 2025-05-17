import logging
import shap

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def explain_prediction_shap(texts: list, model, vectorizer, num_samples: int = 1):
    """
    Generate SHAP values for a list of texts.

    Args:
        texts (list): A list of raw text inputs.
        model: A trained classifier with `predict_proba`.
        vectorizer: A fitted vectorizer that transforms text to input features.
        num_samples (int): Number of samples to explain (optional).

    Returns:
        shap.Explanation object with SHAP values.
    """
    try:
        X = vectorizer.transform(texts)
        explainer = shap.Explainer(model.predict_proba, X)
        shap_values = explainer(X[:num_samples])
        logging.info("✅ SHAP explanation generated.")
        return shap_values
    except Exception as e:
        logging.error(f"SHAP explanation failed: {e}")
        return None
