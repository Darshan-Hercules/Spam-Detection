import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# Import preprocess module
from preprocess import preprocess_data

# Paths setup
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Ensure models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Created models directory: {MODELS_DIR}")

def evaluate_model(model_name, pipeline, X_test, y_test):
    """
    Evaluates a trained pipeline on test data and returns a dictionary of metrics.
    Also plots and saves the confusion matrix heatmap.
    """
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    cm_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"\n=== Evaluation Metrics: {model_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Spam): {prec:.4f}")
    print(f"Recall (Spam):    {rec:.4f}")
    print(f"F1-Score (Spam):  {f1:.4f}")
    print("\nClassification Report:")
    print(report_str)
    
    return {
        'accuracy': acc,
        'precision_spam': prec,
        'recall_spam': rec,
        'f1_spam': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
        'cm_plot_path': cm_path
    }

def extract_top_keywords(pipeline, model_name, top_n=10):
    """
    Extracts the top spam and ham keywords from the vectorizer and classifier coefficients/probabilities.
    """
    tfidf = pipeline.named_steps['tfidf']
    feature_names = np.array(tfidf.get_feature_names_out())
    
    if model_name == "Logistic Regression":
        lr = pipeline.named_steps['lr']
        # Coefficients: positive values point to Spam, negative values point to Ham
        coefs = lr.coef_[0]
        sorted_indices = np.argsort(coefs)
        
        # Most negative are indicative of Ham, most positive are indicative of Spam
        ham_words = [(feature_names[idx], float(coefs[idx])) for idx in sorted_indices[:top_n]]
        spam_words = [(feature_names[idx], float(coefs[idx])) for idx in sorted_indices[-top_n:][::-1]]
        
    elif model_name == "Multinomial Naive Bayes":
        nb = pipeline.named_steps['nb']
        # log probabilities of features given class: feature_log_prob_[0] is Ham, [1] is Spam
        # Relative importance = P(word|Spam) - P(word|Ham)
        log_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
        sorted_indices = np.argsort(log_diff)
        
        # Most negative are indicative of Ham, most positive are indicative of Spam
        ham_words = [(feature_names[idx], float(log_diff[idx])) for idx in sorted_indices[:top_n]]
        spam_words = [(feature_names[idx], float(log_diff[idx])) for idx in sorted_indices[-top_n:][::-1]]
    else:
        ham_words, spam_words = [], []
        
    return {
        'ham_keywords': ham_words,
        'spam_keywords': spam_words
    }

def train_and_evaluate():
    """
    Loads data, trains Naive Bayes and Logistic Regression pipelines, 
    compares them, and saves the best performing one.
    """
    # 1. Load preprocessed data
    df = preprocess_data()
    
    X = df['text']
    y = df['label']
    
    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 3. Define the pipelines
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
        ('nb', MultinomialNB())
    ])
    
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # 4. Train and Evaluate Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    nb_pipeline.fit(X_train, y_train)
    nb_metrics = evaluate_model("Multinomial Naive Bayes", nb_pipeline, X_test, y_test)
    nb_keywords = extract_top_keywords(nb_pipeline, "Multinomial Naive Bayes")
    
    # 5. Train and Evaluate Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)
    lr_keywords = extract_top_keywords(lr_pipeline, "Logistic Regression")
    
    # 6. Compare models based on F1 Score of Spam Class
    print("\n=== Model Selection ===")
    nb_f1 = nb_metrics['f1_spam']
    lr_f1 = lr_metrics['f1_spam']
    
    print(f"Multinomial Naive Bayes Spam F1-Score: {nb_f1:.4f}")
    print(f"Logistic Regression Spam F1-Score:      {lr_f1:.4f}")
    
    if nb_f1 >= lr_f1:
        best_model_name = "Multinomial Naive Bayes"
        best_pipeline = nb_pipeline
        best_metrics = nb_metrics
        best_keywords = nb_keywords
        print(f"\nWinner: {best_model_name} (F1: {nb_f1:.4f})")
    else:
        best_model_name = "Logistic Regression"
        best_pipeline = lr_pipeline
        best_metrics = lr_metrics
        best_keywords = lr_keywords
        print(f"\nWinner: {best_model_name} (F1: {lr_f1:.4f})")
    
    # Save a copy of the best model's confusion matrix as a generic name for Streamlit/Report
    best_cm_img = os.path.join(MODELS_DIR, "best_confusion_matrix.png")
    if os.path.exists(best_metrics['cm_plot_path']):
        import shutil
        shutil.copyfile(best_metrics['cm_plot_path'], best_cm_img)
        print(f"Copied best confusion matrix to {best_cm_img}")
        
    # 7. Fit selected champion model on the entire dataset to maximize training data usage
    print(f"\nRetraining the selected model ({best_model_name}) on the complete dataset...")
    best_pipeline.fit(X, y)
    
    # 8. Save the final pipeline
    model_path = os.path.join(MODELS_DIR, "spam_pipeline.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved the trained pipeline to {model_path}")
    
    # 9. Save metrics and keywords to JSON for app.py and generate_report.py
    results = {
        'best_model': best_model_name,
        'nb_metrics': {k: v for k, v in nb_metrics.items() if k != 'cm_plot_path'},
        'lr_metrics': {k: v for k, v in lr_metrics.items() if k != 'cm_plot_path'},
        'spam_distribution': {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        'ham_keywords': best_keywords['ham_keywords'],
        'spam_keywords': best_keywords['spam_keywords']
    }
    
    results_path = os.path.join(MODELS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved training results metadata to {results_path}")

    # 10. Save JS version of weights for client-side evaluation without server
    tfidf_step = best_pipeline.named_steps['tfidf']
    vocab = tfidf_step.vocabulary_
    vocab_clean = {str(k): int(v) for k, v in vocab.items()}
    idf_weights = tfidf_step.idf_.tolist()
    
    js_export = {
        'model_type': best_model_name,
        'vocabulary': vocab_clean,
        'idf': idf_weights,
        'best_model': best_model_name,
        'nb_metrics': {k: v for k, v in nb_metrics.items() if k != 'cm_plot_path'},
        'lr_metrics': {k: v for k, v in lr_metrics.items() if k != 'cm_plot_path'},
        'spam_distribution': {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        'ham_keywords': best_keywords['ham_keywords'],
        'spam_keywords': best_keywords['spam_keywords']
    }
    
    if best_model_name == "Logistic Regression":
        lr_model = best_pipeline.named_steps['lr']
        js_export['intercept'] = float(lr_model.intercept_[0])
        js_export['coefficients'] = lr_model.coef_[0].tolist()
    elif best_model_name == "Multinomial Naive Bayes":
        nb_model = best_pipeline.named_steps['nb']
        js_export['class_log_prior'] = nb_model.class_log_prior_.tolist()
        js_export['feature_log_prob'] = nb_model.feature_log_prob_.tolist()
        
    js_content = f"// Automatically generated ML Model weights\nconst MODEL_DATA = {json.dumps(js_export, indent=4)};\n"
    js_export_path = os.path.join(SRC_DIR, "model_data.js")
    with open(js_export_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"Saved JavaScript weights to {js_export_path}")

if __name__ == "__main__":
    train_and_evaluate()

