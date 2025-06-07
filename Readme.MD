# Amazon Food Sentiment Analysis

This project performs sentiment analysis on Amazon Fine Food Reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews as expressing positive or negative sentiment based on the review text.

## Overview

The notebook walks through data loading, preprocessing, exploratory data analysis, feature extraction, and classification using machine learning models. It uses Python libraries such as NLTK, pandas, scikit-learn, and matplotlib.

---

## Dataset

The dataset used in this project is the **Amazon Fine Food Reviews** dataset, which contains over 500,000 food-related reviews posted on Amazon. Each entry includes:
- `Text`: The content of the review
- `Score`: The rating (1 to 5 stars)
- `Summary`: A short summary of the review
- `Time`: Review timestamp
- `ProductId`, `UserId`, and other metadata

In this analysis, only the `Text` and `Score` fields are primarily used.

---

## Objective

To simplify the problem into binary sentiment classification:
- Reviews with a score of **4 or 5** are considered **positive**
- Reviews with a score of **1 or 2** are considered **negative**
- Reviews with a score of **3** are **excluded** as they are considered neutral or ambiguous

---

## Steps Performed

### 1. Data Loading
The dataset is loaded using `pandas.read_csv()` and the relevant columns are selected for analysis.

### 2. Data Cleaning
- Duplicate entries are dropped.
- Reviews with a score of 3 are removed.
- A new binary column `Sentiment` is created where:
  - 1 → Positive
  - 0 → Negative

### 3. Text Preprocessing
Performed using **NLTK**:
- Lowercasing all text
- Removing punctuation and non-alphabetic characters
- Tokenization using `nltk.word_tokenize()`
- Stopword removal using `nltk.corpus.stopwords`
- Lemmatization using `WordNetLemmatizer`

This results in clean, tokenized text ready for vectorization.

### 4. Feature Extraction
The cleaned text is transformed into numerical features using:
- **TF-IDF (Term Frequency–Inverse Document Frequency)** via `TfidfVectorizer`
- This captures the importance of words across the corpus while minimizing the impact of frequently used (but uninformative) words

### 5. Model Training
The dataset is split into training and test sets using `train_test_split`.

Machine learning classifiers trained include:
- **Logistic Regression** (used for binary classification due to its simplicity and effectiveness)
- Accuracy and confusion matrix are computed on test data to evaluate performance

---

## Results

- The model achieved reasonable accuracy on the test set, showing it can distinguish between positive and negative sentiments in product reviews.
- TF-IDF proved to be effective for text vectorization.

---

## Dependencies

To run this notebook, you need the following Python libraries:

```bash
pip install pandas numpy matplotlib nltk scikit-learn


   