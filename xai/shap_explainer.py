import logging
import shap
from sklearn.pipeline import make_pipeline
import joblib
import numpy as np
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def explain_prediction_shap(text: str, model, vectorizer, class_names: list, num_features: int = 10) -> list:
    """
    Explain a prediction using SHAP for a single text input.

    Args:
        text (str): The raw input text to explain.
        model: A trained scikit-learn compatible model.
        vectorizer: A fitted vectorizer (TF-IDF, BERT, etc.).
        class_names (list): Names of output classes.
        num_features (int): Number of features to include in the explanation.

    Returns:
        list: List of (word, weight) tuples showing contribution to prediction.
    """

    class_list = ["alt.atheism", " comp.graphics", " comp.os.ms-windows.misc", " comp.sys.ibm.pc.hardware", 
                  " comp.sys.mac.hardware", " comp.windows.x", " misc.forsale", " rec.autos", 
                  " rec.motorcycles", " rec.sport.baseball", " rec.sport.hockey", " sci.crypt", 
                  " sci.electronics", " sci.med", " sci.space", " soc.religion.christian", 
                  " talk.politics.guns", " talk.politics.mideast", " talk.politics.misc", 
                  " talk.religion.misc"]
    
    
    pipeline = make_pipeline(vectorizer, model)
    background_data = np.array(get_balanced_inputs_for_background())
    # background_data = np.array(["testing for atiml"])
    print(type(background_data))
    explainer = shap.KernelExplainer(model.predict_proba, vectorizer.transform(background_data), link='logit')  # text is raw

    instance_numeric = vectorizer.transform([text])
    shap_values = explainer.shap_values(instance_numeric, num_features = 2)# explain same raw text
    
    
    feature_names = vectorizer.get_feature_names_out()
    print(shap_values.shape)

    shap_vals = shap_values[predicted_class][0]  # SHAP values for our instance, for each feature
    top_indices = np.argsort(np.abs(shap_vals))[-num_features:]

# Create a list of tuples: (feature name, corresponding SHAP value)
    input_features = [(feature_names[i], shap_vals[i]) for i in top_indices]

    # Sort the features by absolute SHAP value in descending order
    explanation = sorted(input_features, key=lambda x: abs(x[1]), reverse=True)

    logging.info("✅ SHAP explanation generated.")

    predicted_class = pipeline.predict([text])[0]
    predicted_class_name = class_list[predicted_class]
    logging.info(f"Predicted class:{predicted_class}({predicted_class_name})")
    print(explanation)
    return explanation, predicted_class, predicted_class_name

def get_balanced_inputs_for_background():
    background_list = []
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, "..", "data", "cleaned_20news.csv")
    counter_dict = dict([(i, 0) for i in range(0, 20)])
    with open(file_path, "r") as file:
        for x in file.readlines()[1:]:
            if(counter_dict[int(x.split(",")[1])] != 1):
                background_list.append(x.split(",")[0])
                counter_dict[int(x.split(",")[1])]+=1
    return background_list

def load_saved_model(model_path="../models/final_model.joblib", vectorizer_path="../models/final_vectorizer.joblib"):
    """
    Load trained model and vectorizer from disk.

    Returns:
        model: Classifier
        vectorizer: Corresponding vectorizer
        class_names: List of string class names
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # NOTE: Change this based on your dataset
    class_names = ["alt.atheism", " comp.graphics", " comp.os.ms-windows.misc", " comp.sys.ibm.pc.hardware", " comp.sys.mac.hardware", " comp.windows.x", " misc.forsale", " rec.autos", " rec.motorcycles", " rec.sport.baseball", " rec.sport.hockey", " sci.crypt", " sci.electronics", " sci.med", " sci.space", " soc.religion.christian", " talk.politics.guns", " talk.politics.mideast", " talk.politics.misc", " talk.religion.misc"]

    logging.info("📂 Loaded model and vectorizer from disk")
    return model, vectorizer, class_names

def explain_with_shap():
    text = "final said dream mediterranean new area greater year like holocaust number ist juli usa sweden april still cold chang calendar noth mention true let say true shall azeri women children go pay price rape kill tortur armenian heard someth call geneva convent facist ohhh forgot armenian fight nobodi forgot kill rape tortur kurd turk upon time ohhhh swedish redcross worker lie ever say region killer dont like person shoot that policyl confus search turkish plane dont know talk turkey govern announc give weapon azerbadjan sinc armenia start attack azerbadjan self karabag provinc search plane weapon sinc content announc weapon one that confus that right give weapon azeri sinc armenian start fight azerbadjan shoot armenian bread butter arm personel russian armi"
    model, vectorizer, class_names = load_saved_model()
    shap_values, a, b = explain_prediction_shap(text, model, vectorizer, class_names, 10000)
    # if shap_values is None:
    #     raise HTTPException(status_code=500, detail="SHAP explanation failed.")
    return {
        "shap_values": shap_values.values.tolist(),
        "base_values": shap_values.base_values.tolist(),
        "data": shap_values.data.tolist()
    }

get_balanced_inputs_for_background()
explain_with_shap()