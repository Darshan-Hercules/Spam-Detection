import os
import re
import pandas as pd
import requests

# Constants for file paths and raw dataset URL
DATA_URL = "https://raw.githubusercontent.com/sharmaroshan/Spam-Detection/master/spam.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")

def download_dataset():
    """
    Downloads the SMS Spam Collection dataset if it doesn't already exist in the data folder.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
    
    if not os.path.exists(CSV_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            response = requests.get(DATA_URL, timeout=15)
            response.raise_for_status()
            with open(CSV_PATH, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded dataset to {CSV_PATH}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists at {CSV_PATH}")

def clean_text(text):
    """
    Applies standardization and text cleaning:
    1. Converts text to lowercase.
    2. Removes punctuation, special characters, and numbers.
    3. Removes extra spaces.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation, special characters, and numbers (replace with space to prevent blending words)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove multiple whitespaces and trim leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    """
    Loads raw SMS spam collection data, cleans text, and transforms labels.
    
    Returns:
        pd.DataFrame: Cleaned dataset containing 'label', 'text', 'label_raw', 'text_raw'.
    """
    download_dataset()
    
    # Read CSV. latin-1 encoding is needed as SMS contains non-ASCII characters.
    df = pd.read_csv(CSV_PATH, encoding='latin-1')
    print(f"Raw dataset shape: {df.shape}")
    
    # Clean up unnecessary columns if present (e.g. Unnamed: 2, Unnamed: 3, Unnamed: 4)
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label_raw', 'v2': 'text_raw'})
    else:
        # Fallback if column names are different than Mohit Gupta's CSV
        df = df.iloc[:, [0, 1]]
        df.columns = ['label_raw', 'text_raw']
    
    # Convert labels: ham -> 0, spam -> 1
    df['label'] = df['label_raw'].str.strip().str.lower().map({'ham': 0, 'spam': 1})
    
    # Apply text cleaning
    df['text'] = df['text_raw'].apply(clean_text)
    
    # Drop rows with missing values or empty text resulting from cleaning
    df = df.dropna(subset=['label', 'text'])
    df = df[df['text'] != ""]
    
    print(f"Preprocessed dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")
    
    return df

if __name__ == "__main__":
    # Test script run
    df = preprocess_data()
    print("\nSample preprocessed data:")
    print(df.head())
