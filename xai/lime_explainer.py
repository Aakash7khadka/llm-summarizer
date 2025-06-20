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

    
    try:
        pipeline = make_pipeline(vectorizer, model)
        explainer = LimeTextExplainer(class_names=class_names)

        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=pipeline.predict_proba,
            num_features=num_features
        )

        predicted_class = pipeline.predict([text])[0]
        logging.info(f"Predicted class:{predicted_class}({class_names[predicted_class]})")
        predicted_class_name = class_names[predicted_class]

        logging.info("✅ LIME explanation generated.")
        # prompt_for_ollamaprediction(explanation.as_list())
        return explanation.as_list(), predicted_class, predicted_class_name

    except Exception as e:
        logging.error(f"LIME explanation failed: {e}")
        return []



def explain_classification_decision_with_ollama(text, predicted_class_index, predicted_class_name):
    # class_list = ["alt.atheism", " comp.graphics", " comp.os.ms-windows.misc", " comp.sys.ibm.pc.hardware", " comp.sys.mac.hardware", " comp.windows.x", " misc.forsale", " rec.autos", " rec.motorcycles", " rec.sport.baseball", " rec.sport.hockey", " sci.crypt", " sci.electronics", " sci.med", " sci.space", " soc.religion.christian", " talk.politics.guns", " talk.politics.mideast", " talk.politics.misc", " talk.religion.misc"]
    # predicted_class = class_list[predicted_class_index]
    prompt = """
            Following text has been classified as """ +predicted_class_name+ """
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

         Stemmed Input Text: 
            """ + text +"""
                Explain the classification decision why the class was chosen
            """
    response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        "model": "gemma3:1b",
        "prompt": prompt,
        "stream": False,  # Set to True for streamed response
        "context":[]
    }
    )

    # Print the generated response

    resp = response.json()["response"]
    logging.info("Ollama Explanation about classification:")
    logging.info(response.json()["response"])   
    return resp


def ollamaprediction_comparison_with_limeexplanation(text, ollama_explanation, explanation_list, predicted_class_name):
    # Convert LIME explanation to readable format
    readable_expl = "\n".join([f"- {word}: {score:.4f}" for word, score in explanation_list])

    categories = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos",
    "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
    "sci.electronics", "sci.med", "sci.space", "soc.religion.christian",
    "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"
]
    valid_list = "\\n".join(f"- {c}" for c in categories)

    lime_prompt = f"""
    You are a classification analyst for 20-Newsgroups data.

    You are given:
    - Stemmed text
    - The category predicted by a ML model for the text
    - The original LLM explanation for the prediction for the text input
    - A LIME explanation showing important words and their contribution

    ---
    **Text:** {text}

    **Predicted category:** {predicted_class_name}

    **Original LLM explanation:**  
    \"\"\"{ollama_explanation}\"\"\"

    

    **LIME word importance:**
    {readable_expl}

    ---

        Your task is to answer the following **question**, using **only this format**:

        **Does the LIME explanation support, or contradict the Original LLM explanation?**  
        (Choose exactly one: **support** or **contradict**)

        - If your answer is **support**, explain briefly why.
        - If your answer is **contradict**, explain briefly why, and you **must suggest a new category** from the allowed list below.

        **⚠️ You must always suggest a new category if you answer "contradict".**

        **Allowed categories only:**  
        {valid_list}

        ---

        💡 Your answers must be logically consistent.
        - You must not invent a category outside the list.
        - You must follow the required format exactly.
    """


    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "gemma3:1b",
            "prompt": lime_prompt,
            "stream": False,
            "temperature": 0.2,
            "context": []
        }
    )
    logging.info("Ollama explanation with LIME values and its own classification decision:")
    logging.info(response.json()["response"])
    return response.json()["response"]
