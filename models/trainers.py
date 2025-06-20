import logging
import joblib
from typing import Union, Tuple
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

# Setup logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def get_model(model_type: str = "svm", **kwargs) -> ClassifierMixin:
    """
    Create and return an untrained model based on the specified type.

    Args:
        model_type (str): One of ['svm', 'mlp', 'rf']
        kwargs: Optional model-specific hyperparameters

    Returns:
        Untrained scikit-learn model
    """
    model_type = model_type.lower()
    logging.info(f"📦 Initializing model: {model_type.upper()}")

    if model_type == "svm":
        return SVC(kernel='linear', probability=True, random_state=42, **kwargs)
    elif model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, **kwargs)
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"❌ Invalid model_type: {model_type}. Choose from 'svm', 'mlp', or 'rf'.")


def train_model(X_train, y_train, model_type: str = "svm", **kwargs) -> ClassifierMixin:
    """
    Train a model on the given data.

    Args:
        X_train: Feature matrix
        y_train: Labels
        model_type (str): Type of model to train
        kwargs: Additional hyperparameters

    Returns:
        Trained model
    """
    model = get_model(model_type, **kwargs)
    logging.info(f"🧠 Training {model_type.upper()} model...")
    model.fit(X_train, y_train)
    logging.info(f"✅ {model_type.upper()} training complete")
    return model

'''
def save_model(model, vectorizer, model_path="models/final_model.joblib", vectorizer_path="models/final_vectorizer.joblib"):
    """
    Save trained model and vectorizer to disk.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    logging.info(f"💾 Model saved to {model_path}")
    logging.info(f"💾 Vectorizer saved to {vectorizer_path}")
'''

def save_model(model, model_path="models/final_model.joblib", vectorizer_path="models/final_vectorizer.joblib"):
    """
    Save trained model to disk.
    """
    joblib.dump(model, model_path)
    logging.info(f"💾 Model saved to {model_path}")
    #logging.info(f"💾 Vectorizer saved to {vectorizer_path}")

def load_saved_model(model_path="models/final_model.joblib", vectorizer_path="models/final_vectorizer.joblib") -> Tuple:
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
