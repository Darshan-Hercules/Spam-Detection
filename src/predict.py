import os
import joblib

# Paths setup
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "spam_pipeline.pkl")

# Cached model instance to prevent redundant disk I/O on repeated calls
_model = None

def load_model():
    """
    Loads the trained model pipeline from the pickle file, caching it in memory.
    
    Returns:
        Pipeline: The Scikit-Learn Pipeline object containing TF-IDF and the classifier.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run 'python src/train.py' first to train and save the model."
            )
        print(f"Loading trained pipeline from {MODEL_PATH}...")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_spam(message):
    """
    Predicts if a message is Spam or Ham.
    
    Args:
        message (str): The raw text message to classify.
        
    Returns:
        str: "Spam" if the model classifies the text as spam, else "Ham".
    """
    # Handle empty/whitespace inputs
    if not message or not isinstance(message, str) or not message.strip():
        return "Ham"
    
    try:
        pipeline = load_model()
        # The pipeline expects a list/array of text strings as input.
        prediction = pipeline.predict([message])[0]
        return "Spam" if prediction == 1 else "Ham"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Ham" # Default safe classification

def predict_spam_with_confidence(message):
    """
    Predicts if a message is Spam or Ham along with probability confidence.
    
    Args:
        message (str): The raw text message.
        
    Returns:
        tuple: (str, float) -> (Prediction: "Spam"/"Ham", Confidence: float [0.0 - 1.0])
    """
    if not message or not isinstance(message, str) or not message.strip():
        return "Ham", 1.0
        
    try:
        pipeline = load_model()
        prediction = pipeline.predict([message])[0]
        probabilities = pipeline.predict_proba([message])[0]
        
        # probabilities has index 0 as Ham and index 1 as Spam
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        label = "Spam" if prediction == 1 else "Ham"
        
        return label, float(confidence)
    except Exception as e:
        print(f"Error during prediction with confidence: {e}")
        return "Ham", 0.0

if __name__ == "__main__":
    import sys
    
    # Check if a custom message was passed as a command-line argument
    if len(sys.argv) > 1:
        test_message = " ".join(sys.argv[1:])
    else:
        # Default test message (obvious spam)
        test_message = "URGENT! Your mobile number has been selected for a £2,000 prize. Call 09061701461 to claim. T&C Apply."
        print("No message provided as argument. Running with default spam example...")
        
    print(f"\nEvaluating message:\n\"{test_message}\"")
    
    try:
        label, conf = predict_spam_with_confidence(test_message)
        print("-" * 40)
        print(f"Classification Result : {label}")
        print(f"Model Confidence      : {conf * 100:.2f}%")
        print("-" * 40)
    except Exception as e:
        print(f"Failed to execute prediction: {e}")
