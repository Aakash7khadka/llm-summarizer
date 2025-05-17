import logging
from lime.lime_text import LimeTextExplainer
import numpy as np
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline


def explain_prediction_lime(text, model, vectorizer, class_names):
    """
    Generate a LIME explanation for a single prediction.
    """
    pipeline = make_pipeline(vectorizer, model)
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text, pipeline.predict_proba, num_features=10)
    return exp.as_list()


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def explain_prediction_lime(text: str, model, vectorizer, class_names: list, num_features: int = 10) -> list:
    """
    Explain a prediction using LIME for a single text input.

    Args:
        text (str): The raw input text to explain.
        model: A trained scikit-learn compatible model.
        vectorizer: A fitted vectorizer (TF-IDF, BERT, etc.).
        class_names (list): Names of output classes.
        num_features (int): Number of features to include in the explanation.

    Returns:
        list: List of (word, weight) tuples showing contribution to prediction.
    """
    try:
        pipeline = make_pipeline(vectorizer, model)
        explainer = LimeTextExplainer(class_names=class_names)

        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=pipeline.predict_proba,
            num_features=num_features
        )

        logging.info("✅ LIME explanation generated.")
        return explanation.as_list()

    except Exception as e:
        logging.error(f"LIME explanation failed: {e}")
        return []
