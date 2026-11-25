"""Test if the fixes resolved the class imbalance issue."""

import pandas as pd
import joblib
from pathlib import Path
from utils import clean_text, label_sentiment

print("=" * 70)
print("TESTING ML PIPELINE FIXES")
print("=" * 70)

# Test 1: Check stopwords don't include negation words
print("\n1. VERIFYING STOPWORDS FIX:")
from utils import STOPWORDS

critical_words = ['not', 'no', 'never', 'nor', 'neither', 'dont', "don't"]
for word in critical_words:
    status = "❌ STILL IN STOPWORDS" if word in STOPWORDS else "✓ REMOVED"
    print(f"   '{word}': {status}")

# Test 2: Check sentiment keyword expansion
print("\n2. CHECKING SENTIMENT KEYWORDS:")
from utils import POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS
print(f"   Positive keywords: {len(POSITIVE_KEYWORDS)} words")
print(f"   Negative keywords: {len(NEGATIVE_KEYWORDS)} words")

# Test 3: Test label_sentiment with critical reviews
print("\n3. TESTING LABEL_SENTIMENT FUNCTION:")
test_cases = [
    ("This is excellent service", "Positive"),
    ("This is bad service", "Negative"),
    ("This is not bad", "Negative"),  # Critical: should detect "not bad"
    ("not good at all", "Negative"),
    ("This is average", "Neutral"),
]

for review, expected in test_cases:
    label = label_sentiment(review)
    status = "✓" if label == expected else f"❌ (got {label})"
    print(f"   '{review}' -> {label} {status}")

# Test 4: Verify data exists for training
print("\n4. CHECKING DATA:")
csv_path = Path("processed/cleaned_reviews.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"   Dataset size: {len(df)} reviews")
    print(f"   Class distribution:")
    for sent, count in df['Sentiment'].value_counts().items():
        pct = count / len(df) * 100
        print(f"     {sent}: {count} ({pct:.1f}%)")
else:
    print(f"   ❌ No dataset found. Run: python run_pipeline.py")

# Test 5: Check if models exist
print("\n5. CHECKING TRAINED MODELS:")
models_dir = Path("models")
for model_name in ["logistic.pkl", "naive_bayes.pkl", "random_forest.pkl", "vectorizer.pkl"]:
    path = models_dir / model_name
    status = "✓ EXISTS" if path.exists() else "❌ NOT FOUND"
    print(f"   {model_name}: {status}")

print("\n" + "=" * 70)
print("If all tests pass, run: python run_pipeline.py")
print("=" * 70)
