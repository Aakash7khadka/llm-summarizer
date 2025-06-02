import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# from data.data_loader import load_cleaned_20news  # Or load_cleaned_agnews
from vectorization.vectorizers import vectorize_tfidf
from models.trainers import train_model
from evaluation.metrics import evaluate_model, plot_confusion_matrix
from models.training import training_process

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def main():
    # Load and split data
    logging.info("🚀 Starting model training pipeline...")

    df = pd.read_csv("data/cleaned_20news.csv")  # Or load_cleaned_20news()
    texts, labels = df['text'].tolist(), df['label'].tolist()

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    # Vectorize with TF-IDF
    X_train_vecs, vectorizer = vectorize_tfidf(X_train_texts)
    X_test_vecs = vectorizer.transform(X_test_texts)

    #train and evaluate models
    #training_process (None)

    # Train and evaluate models
    model_types = ["svm", "mlp", "rf"]
    for model_type in model_types:
        logging.info(f"🔧 Training model: {model_type.upper()}")

        model = train_model(X_train_vecs, y_train, model_type=model_type)
        y_pred = model.predict(X_test_vecs)

        metrics = evaluate_model(y_test, y_pred)
        logging.info(f"📊 {model_type.upper()} metrics: {metrics}")

        # Optional: Save confusion matrix
        cm_path = f"outputs/confusion_matrix_{model_type}.png"
        plot_confusion_matrix(y_test, y_pred, labels=sorted(
            set(labels)), save_path=cm_path, show=False)


if __name__ == "__main__":
    main()
