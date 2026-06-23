# SMS Spam Detection System

A complete machine learning project that classifies SMS text messages as **Spam** or **Ham (Legitimate)**. The project compares **Multinomial Naive Bayes** and **Logistic Regression**, builds a production-grade Scikit-Learn pipeline, and deploys an interactive **Streamlit dashboard** with dynamic data analytics. It also includes automated data setup and generates an internship-ready PDF report.

---

## 📂 Project Structure

```
SpamDetection/
│
├── data/
│   └── spam.csv                # Raw dataset (downloaded automatically)
│
├── models/
│   ├── spam_pipeline.pkl       # Saved Scikit-Learn pipeline (TF-IDF + Best Model)
│   ├── results.json            # Model metrics, class count, and top vocabulary keywords
│   ├── best_confusion_matrix.png # Confusion matrix plot of the winning model
│   └── label_distribution.png  # Bar chart of dataset categories
│
├── src/
│   ├── preprocess.py           # Dataset acquisition, columns cleaning, and text standardization
│   ├── train.py                # Model training, comparative evaluations, and serialization
│   ├── predict.py              # CLI and API inference functions with confidence estimation
│   ├── generate_report.py      # Automated compilation of report.pdf via reportlab
│   └── app.py                  # Streamlit visual web dashboard
│
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation (This file)
└── report.pdf                  # Internship project report (generated programmatically)
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Clone or copy the project files to your local folder. Open your command terminal, navigate to the `SpamDetection` folder, and execute:
```bash
pip install -r requirements.txt
```
This will install all necessary libraries, including `pandas`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `seaborn`, and `reportlab`.

---

## 🚀 Execution Workflow

The system is fully automated. You can train the models, compile the PDF, and launch the web app with these steps:

### Step 1: Run Model Training & Evaluation
Run the training script to download the dataset, preprocess it, compare Naive Bayes and Logistic Regression, and save the best-performing pipeline:
```bash
python src/train.py
```
This script will output evaluation results to the console, save the winning pipeline to `models/spam_pipeline.pkl`, and output metrics metadata to `models/results.json`.

### Step 2: Generate the PDF Report
To compile the professional multi-page internship report, run:
```bash
python src/generate_report.py
```
This will build a beautifully structured report named `report.pdf` at the root directory of the project.

### Step 3: Test CLI Inference (Optional)
You can quickly run predictions on custom messages directly from your command line:
```bash
python src/predict.py "URGENT! You have won a £2,000 cash prize. Call 09061701461 now to claim."
```

### Step 4: Launch the Streamlit Web Application
Launch the graphical dashboard to interactively test messages and explore model details:
```bash
streamlit run src/app.py
```
The application will open automatically in your browser (usually at `http://localhost:8501`).

---

## 📘 Mathematical & Conceptual Foundations

### 1. Text Preprocessing Pipeline
To transform raw human-written text into structured inputs, we apply a series of standard text normalization processes:
- **Lowercase Conversion:** Matches terms regardless of capitalization (e.g., `"FREE"` and `"free"` map to the same token).
- **Character Filtering:** Filters out numbers, punctuation, and non-alphabetic special symbols using the regular expression `[^a-zA-Z\s]`. This strips formatting noise while keeping semantic words.
- **Whitespace Collapsing:** Replaces multi-space blocks with single spaces and strips outer padding.
- **Label Mapping:** Converts the target string categories into numeric classification indices:
  $$\text{Ham} \rightarrow 0$$
  $$\text{Spam} \rightarrow 1$$

### 2. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF is a statistical weighting scheme representing how important a word is to a document in a collection. It consists of two components:
1. **Term Frequency (TF):** Measures how frequently a term $t$ appears in document $d$:
   $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
2. **Inverse Document Frequency (IDF):** Measures how common or rare a term is across all documents in the corpus $D$:
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   *(We use Scikit-Learn's smoothed IDF formulation).*
3. **TF-IDF Weight:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
This suppresses high-frequency grammatical words (like "and", "the") while boosting specific vocabulary cues (like "claim", "prize", "txt").

### 3. Machine Learning Classifiers

#### Multinomial Naive Bayes (MNB)
Naive Bayes is a classification technique based on Bayes' Theorem, which models the conditional probability of class $C_k$ (Ham or Spam) given an input feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ representing word TF-IDF weights:
$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}$$
It assumes conditional independence between features given the class (the "naive" assumption):
$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^n P(x_i \mid C_k)$$
Taking the log for numerical stability:
$$\log P(C_k \mid \mathbf{x}) \propto \log P(C_k) + \sum_{i=1}^n \log P(x_i \mid C_k)$$
For text inputs, **Multinomial Naive Bayes** represents counts or weights from a multinomial distribution.

#### Logistic Regression
Logistic Regression is a linear classifier that computes the probability of a binary label using the log-odds format. The input features are multiplied by coefficients (weights) $W$ and added to a bias term $b$, which is then fed into the Logistic (Sigmoid) function:
$$P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- If $\sigma(\mathbf{w}^T \mathbf{x} + b) \geq 0.5$, classification is **Spam (1)**.
- If $\sigma(\mathbf{w}^T \mathbf{x} + b) < 0.5$, classification is **Ham (0)**.
Weights are optimized during training by minimizing the Binary Cross-Entropy Loss.

---

## 📊 Evaluation Metrics

Since spam datasets are typically highly imbalanced, relying on overall **Accuracy** can be misleading (a naive model predicting all messages as "Ham" would still achieve high accuracy). Therefore, we focus on Spam class (label 1) specific metrics:

- **Accuracy:** The ratio of correct predictions to total cases:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision:** The ratio of true spam detections to all messages *flagged* as spam. High precision guarantees legitimate messages (ham) are not accidentally routed to spam folders (minimizing false alarms):
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity):** The ratio of true spam detections to all *actual* spam messages. High recall ensures most malicious messages are caught (minimizing spam leakage):
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1 Score:** The harmonic mean of Precision and Recall, which is the standard optimization target to balance detection rates and false alarm rates:
  $$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 🛠️ Modularity and Code Quality
- **Decoupled Architecture:** Preprocessing, training, prediction, report compilation, and UI dashboards are separated into clean individual scripts.
- **Error Handling:** Fallbacks are implemented for missing files, invalid input queries, and network timeout during dataset acquisition.
- **Scikit-Learn Pipeline:** The TF-IDF Vectorizer and model are compiled into a single pipeline object, guaranteeing that training text cleaning transforms match inference text transforms exactly, preventing data leakage.
