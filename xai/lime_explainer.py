import logging
from lime.lime_text import LimeTextExplainer
import numpy as np
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
import requests


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

    class_list = ["alt.atheism", " comp.graphics", " comp.os.ms-windows.misc", " comp.sys.ibm.pc.hardware", " comp.sys.mac.hardware", " comp.windows.x", " misc.forsale", " rec.autos", " rec.motorcycles", " rec.sport.baseball", " rec.sport.hockey", " sci.crypt", " sci.electronics", " sci.med", " sci.space", " soc.religion.christian", " talk.politics.guns", " talk.politics.mideast", " talk.politics.misc", " talk.religion.misc"]
    try:
        pipeline = make_pipeline(vectorizer, model)
        explainer = LimeTextExplainer(class_names=class_names)

        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=pipeline.predict_proba,
            num_features=num_features
        )

        predicted_class = pipeline.predict([text])[0]
        logging.info(f"Predicted class:{predicted_class}({class_list[predicted_class]})")

        logging.info("✅ LIME explanation generated.")
        # prompt_for_ollamaprediction(explanation.as_list())
        return explanation.as_list(), predicted_class

    except Exception as e:
        logging.error(f"LIME explanation failed: {e}")
        return []


def explain_classification_decision(text, predicted_class_index):
    class_list = ["alt.atheism", " comp.graphics", " comp.os.ms-windows.misc", " comp.sys.ibm.pc.hardware", " comp.sys.mac.hardware", " comp.windows.x", " misc.forsale", " rec.autos", " rec.motorcycles", " rec.sport.baseball", " rec.sport.hockey", " sci.crypt", " sci.electronics", " sci.med", " sci.space", " soc.religion.christian", " talk.politics.guns", " talk.politics.mideast", " talk.politics.misc", " talk.religion.misc"]
    predicted_class = class_list[predicted_class_index]
    prompt = """
            Following text has been classified as """ +predicted_class+ """
            among the following class labels:
            - alt.atheism
            - comp.graphics
            - comp.os.ms-windows.misc
            - comp.sys.ibm.pc.hardware
            - comp.sys.mac.hardware
            - comp.windows.x
            - misc.forsale
            - rec.autos
            - rec.motorcycles
            - rec.sport.baseball
            - rec.sport.hockey
            - sci.crypt
            - sci.electronics
            - sci.med
            - sci.space
            - soc.religion.christian
            - talk.politics.guns
            - talk.politics.mideast
            - talk.politics.misc
            - talk.religion.misc

            For the text: 
            """ + text +"""
                Explain the classification decision in very short
            """
    response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        "model": "gemma3:1b",
        "prompt": prompt,
        "stream": False  # Set to True for streamed response
    }
    )

    # Print the generated response

    resp = response.json()["response"]
    logging.info(response.json()["response"])   
    return resp


def ollamaprediction_comparison_limeexplanation(text, ollama_explanation, explanation_list):
    # Convert LIME explanation to readable format
    readable_expl = "\n".join([f"- {word}: {score:.4f}" for word, score in explanation_list])

    # Category list to give context
    categories = """
    List of valid categories:
    - alt.atheism
    - comp.graphics
    - comp.os.ms-windows.misc
    - comp.sys.ibm.pc.hardware
    - comp.sys.mac.hardware
    - comp.windows.x
    - misc.forsale
    - rec.autos
    - rec.motorcycles
    - rec.sport.baseball
    - rec.sport.hockey
    - sci.crypt
    - sci.electronics
    - sci.med
    - sci.space
    - soc.religion.christian
    - talk.politics.guns
    - talk.politics.mideast
    - talk.politics.misc
    - talk.religion.misc
    """

    lime_prompt = f"""
You are analyzing a classification decision made for a document from the 20 Newsgroups dataset.

Step 1: Input Text:
\"\"\"{text}\"\"\"

Step 2: Your previous LLM explanation and predicted category:
\"\"\"{ollama_explanation}\"\"\"

Step 3: Now, here is a LIME explanation of the model’s decision. It shows how each word influenced the prediction (positive: supports the category, negative: opposes it):

LIME Explanation:
{readable_expl}

{categories}

Now answer briefly:
1. Does the LIME explanation support, contradict, or refine your earlier explanation?
2. Mention 1–2 specific words from the LIME output and explain their role.
3. Would you change your original predicted category or explanation? Yes or No — and why.
(Only pick from the category list above.)
"""

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "gemma3:1b",
            "prompt": lime_prompt,
            "stream": False,
            "temperature": 0.2
        }
    )

    logging.info(response.json()["response"])
    return response.json()["response"]
