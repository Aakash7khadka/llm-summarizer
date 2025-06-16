from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from data.text_cleaner import clean_text
from llm.bert_summarizer import generate_bert_summary
from llm.ollama_client import generate_llm_summary
from vectorization.vectorizers import vectorize_tfidf, vectorize_bert
from llm.summarize_and_vectorize import summarize_and_vectorize
from xai.lime_explainer import explain_prediction_lime, explain_classification_decision_with_ollama, ollamaprediction_comparison_with_limeexplanation

from xai.shap_explainer import explain_prediction_shap
from models.trainers import load_saved_model

app = FastAPI(
    title="Text Summarization & Vectorization API",
    description="Exposes endpoints for preprocessing, summarization, embedding, and explainability (LIME/SHAP).",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Text Processing API is running 🚀"}


# --- Preprocessing ---
@app.post("/preprocess")
def preprocess_text(request: TextRequest):
    cleaned = clean_text(request.text)
    return {"cleaned_text": cleaned}


# --- Summarization ---
@app.post("/summarize/bert")
def summarize_bert(request: TextRequest):
    summary = generate_bert_summary(request.text)
    if not summary:
        raise HTTPException(
            status_code=400, detail="BERT summarization failed.")
    return {"bert_summary": summary}


@app.post("/summarize/llm")
def summarize_llm(request: TextRequest):
    summary = generate_llm_summary(request.text)
    if not summary:
        raise HTTPException(
            status_code=400, detail="LLM summarization failed.")
    return {"llm_summary": summary}


# --- Vectorization ---
@app.post("/vectorize/tfidf")
def vectorize_tfidf_endpoint(request: TextRequest):
    vectors, _ = vectorize_tfidf([request.text])
    return {"tfidf_vector": vectors.toarray()[0].tolist()}


@app.post("/vectorize/bert")
def vectorize_bert_endpoint(request: TextRequest):
    vectors, _ = vectorize_bert([request.text])
    return {"bert_vector": vectors[0]}


# --- Combined pipeline ---
@app.post("/summarize-and-vectorize")
def summarize_and_vectorize_endpoint(request: TextRequest):
    result = summarize_and_vectorize(request.text)
    return result


# --- Explainability ---
@app.post("/xai/lime")
def explain_with_lime(request: TextRequest):
    model, vectorizer, class_names = load_saved_model()
    explanation, predicted_class, predicted_class_name = explain_prediction_lime(
        request.text, model, vectorizer, class_names, 10000)
    llm_reasoning = explain_classification_decision_with_ollama(request.text, predicted_class, predicted_class_name)
    ollamaprediction_comparison_with_limeexplanation(request.text, llm_reasoning, explanation, predicted_class_name)
    return {"lime_explanation": explanation}


@app.post("/xai/shap")
def explain_with_shap(request: TextRequest):
    model, vectorizer, class_names = load_saved_model()
    shap_values,a , b = explain_prediction_shap([request.text], model, vectorizer, class_names, 10000)
    if shap_values is None:
        raise HTTPException(status_code=500, detail="SHAP explanation failed.")
    return {
        "shap_values": shap_values.values.tolist(),
        "base_values": shap_values.base_values.tolist(),
        "data": shap_values.data.tolist()
    }
