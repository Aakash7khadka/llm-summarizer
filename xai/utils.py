
import joblib
import logging
from typing import  Tuple
from pydantic import BaseModel
import requests
import pickle
import json
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import InconsistentVersionWarning



def load_saved_model(model_path="../models/final_model.joblib") -> Tuple:
    """
    Load trained model and vectorizer from disk.

    Returns:
        model: Classifier
        vectorizer: Corresponding vectorizer
        class_names: List of string class names
    """
    model = joblib.load(model_path)

    # NOTE: Change this based on your dataset
    class_names = ["alt.atheism", " comp.graphics", " comp.os.ms-windows.misc", " comp.sys.ibm.pc.hardware", " comp.sys.mac.hardware", " comp.windows.x", " misc.forsale", " rec.autos", " rec.motorcycles", " rec.sport.baseball", " rec.sport.hockey", " sci.crypt", " sci.electronics", " sci.med", " sci.space", " soc.religion.christian", " talk.politics.guns", " talk.politics.mideast", " talk.politics.misc", " talk.religion.misc"]

    logging.info("📂 Loaded model from disk")
    return model, class_names


def load_saved_vectorizer(vectorizer_path) -> Tuple:
    """
    Load trained model and vectorizer from disk.

    Returns:
        model: Classifier
        vectorizer: Corresponding vectorizer
        class_names: List of string class names
    """


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
    vectorizer = joblib.load(vectorizer_path)
   
    logging.info("📂 Loaded vectorizer from disk")
    return  vectorizer



class RequestDataLime(BaseModel):
    line_id: str 
    useSummaryAlso: bool


def get_data_by_lineid(line_id):
    logging.info("===============================================================================")
    with open("data/cleaned_20news.csv", "r") as file:
        line = file.readlines()[int(line_id)]
    text = line.split(",")[0]
    actual_label_class = line.split(",")[1]
    actual_label = line.split(",")[2]
    logging.info(f"➡️ Actual class is:{actual_label_class}({actual_label.replace("\n","")})")
    return text, actual_label_class, actual_label

def get_summary_text_by_lineid(line_id):
    with open("data/summaries_20news_lsa.json", "r") as file:
        summary_dict = json.load(file)
        line_id = str(int(line_id)-1)
        print(line_id)
        text = summary_dict[line_id]
    return text

def collect_incorrect_classifications():
    model, class_names = load_saved_model("models/mlp_tfidf_whole.joblib")
    vectorizer = load_saved_vectorizer(vectorizer_path="vectorization/tfidf_vectorizer_full.pkl")
    with open("data/cleaned_20news.csv", "r") as file:
        lines = file.readlines()[1:1000]
    for i, line in enumerate(lines):
        text = line.split(",")[0]
        actual_label_class = line.split(",")[1]
        actual_label = line.split(",")[2]
        transformed_text = vectorizer.transform([text])
        a= str(model.predict(transformed_text)[0]).strip()
        b= str(actual_label_class.strip())
        if(a != b):
            print(f">>{i+1}")
            print("Predicted: "+str(model.predict(transformed_text)[0]))
            print("Actual: "+actual_label_class)
      



# collect_incorrect_classifications()