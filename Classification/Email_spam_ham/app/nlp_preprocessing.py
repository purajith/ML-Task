import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def nlp_preprocessing(df):
    # Label encoding
    df["Category"].replace({"ham": 0, "spam": 1}, inplace=True)

    # Convert all text to lowercase
    df["Message"] = df["Message"].str.lower()

    # Normalize white space and remove punctuation
    df["cleaning_text"] = df["Message"].apply(lambda x: re.sub(r'[^\w\s]', '', re.sub(r'\s+', ' ', x).strip()))

    # Tokenization
    df["cleaning_text"] = df["cleaning_text"].apply(word_tokenize)

    # Remove stopwords
    df["cleaning_text"] = df["cleaning_text"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

    # Join tokens back to string
    df["cleaning_text"] = df["cleaning_text"].apply(lambda tokens: " ".join(tokens))

    return df

if __name__ == "__main__":
    data_path = "data/email.csv"
    df = pd.read_csv(data_path)
    df = nlp_preprocessing(df)
    print(df.head())
    # Save processed data
    # df.to_csv("data/processed_data.csv", index=False, encoding='utf-8')
    print("[INFO] Preprocessing complete. Sample data:\n", df.head())
