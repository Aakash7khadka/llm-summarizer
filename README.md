# Adjusted README content with improved formatting and clearer explanation

adjusted_readme = """
# 🧠 LLM Summary-Based Document Classification

This project investigates whether LLM-generated summaries (via Gemma 2B using Ollama) can effectively replace full-text document representations for classification tasks. We compare four different approaches to representing documents:

- TF-IDF vectors
- Doc2Vec embeddings
- Extractive summaries (using TextRank)
- Abstractive summaries (via locally running LLM)

Datasets used:
- **20 Newsgroups** (mandatory)
- **AG News** (used for few-shot/zero-shot experiments)

---

## 📁 Project Structure

```bash
llm_summary_classification/
│
├── data/                         # All data files
│   ├── raw/                      # Original datasets
│   ├── cleaned_20news.csv        # Cleaned version of 20 Newsgroups
│   ├── cleaned_agnews.csv        # Cleaned version of AG News
│   ├── summaries_llm.json        # Abstractive summaries via LLM
│   └── summaries_textrank.json   # Extractive summaries via TextRank
│
├── utils/                        # Reusable Python modules (API logic)
│   ├── __init__.py
│   ├── data_loader.py            # Loading, cleaning, and saving datasets
│   ├── summary_api.py            # Summary generation via LLM and TextRank
│   ├── vectorization_api.py      # TF-IDF, Doc2Vec, and BERT encodings
│   ├── modeling.py               # SVM, MLP, RF training functions
│   └── evaluation.py             # Metrics calculation and plotting
│
├── notebooks/                    # Optional test notebooks
│   ├── test_vectorization.ipynb
│   └── test_textrank.ipynb
│
├── experiments/                  # Scripts to run training and tests
│   ├── run_all_models.py         # Runs classifiers on all vector types
│   └── few_shot_test.py          # Few-shot/zero-shot scenario evaluation
│
├── outputs/                      # Store results and evaluation plots
│   ├── metrics_20news.json
│   └── plots/
│       ├── confusion_svm.png
│       └── runtime_plot.png
│
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation (this file)
└── project_plan.pdf              # Full technical project plan
```
##  Key Files to Know

- **`data_loader.py`**  
  Loads and cleans both datasets (20 Newsgroups and AG News). Prepares them for summarization and vectorization.

- **`summary_api.py`**  
  Generates summaries using:
  - **Gemma 2B** (via Ollama) for abstractive summarization  
  - **TextRank** (via `sumy`) for extractive summarization

- **`vectorization_api.py`**  
  Converts documents or summaries into vector form using:
  - **TF-IDF** (`scikit-learn`)  
  - **Doc2Vec** (`gensim`)  
  - **Sentence-BERT** (`sentence-transformers`)

- **`modeling.py`**  
  Trains and tests classification models:
  - Support Vector Machine (SVM)  
  - Multi-Layer Perceptron (MLP)  
  - Random Forest

- **`evaluation.py`**  
  Computes classification metrics:
  - Accuracy, Precision, Recall, F1-score  
  - Generates confusion matrices and runtime plots

- **`run_all_models.py`**  
  Executes the full ML pipeline on all document representations and saves evaluation results for comparison.
