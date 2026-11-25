"""Diagnostic script to debug the ML pipeline issues."""

import pandas as pd
import json
from pathlib import Path
from utils import clean_text, label_sentiment

PROJECT_ROOT = Path(__file__).parent.resolve()
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 70)
print("DIAGNOSTIC REPORT: ML PIPELINE ISSUE")
print("=" * 70)

# Check if cleaned dataset exists
csv_path = PROCESSED_DIR / "cleaned_reviews.csv"
if not csv_path.exists():
    print("\n❌ ERROR: No cleaned dataset found!")
    print(f"Expected: {csv_path}")
    print("\nRun: python run_pipeline.py")
    exit(1)

# Load dataset
df = pd.read_csv(csv_path)
print(f"\n1. DATASET STATISTICS")
print(f"   Total reviews: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Check class distribution
print(f"\n2. CLASS DISTRIBUTION")
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)
print(f"\n   Percentages:")
for sentiment, count in sentiment_counts.items():
    pct = (count / len(df)) * 100
    print(f"   {sentiment}: {pct:.1f}%")

# Check for class imbalance
max_class = sentiment_counts.max()
min_class = sentiment_counts.min()
imbalance_ratio = max_class / min_class
print(f"\n   Imbalance Ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 3:
    print(f"   ⚠️  WARNING: Severe imbalance detected!")

# Sample reviews by class
print(f"\n3. SAMPLE REVIEWS BY CLASS")
for sentiment in ['Positive', 'Negative', 'Neutral']:
    samples = df[df['Sentiment'] == sentiment]['Review'].head(2).tolist()
    print(f"\n   {sentiment}:")
    for i, review in enumerate(samples, 1):
        print(f"     [{i}] {review[:60]}...")

# Test label_sentiment function
print(f"\n4. TESTING LABEL_SENTIMENT FUNCTION")
test_cases = [
    "excellent service and very fast",
    "terrible quality very bad",
    "not good at all",
    "this is okay",
    "worst experience ever",
    "amazing and perfect",
]

for review in test_cases:
    label = label_sentiment(review)
    print(f"   {review:40} -> {label}")

# Check cleaned text loss
print(f"\n5. TEXT PREPROCESSING CHECK")
examples = [
    "This is NOT good service",
    "Terrible and bad quality",
    "Excellent service",
]

for text in examples:
    cleaned = clean_text(text)
    print(f"   Original:  {text}")
    print(f"   Cleaned:   {cleaned}")
    print()

# Check metrics if available
metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    print(f"\n6. CURRENT MODEL METRICS")
    with open(metrics_path) as f:
        metrics = json.load(f)
        for model, scores in metrics.items():
            print(f"   {model}:")
            print(f"     Accuracy: {scores['accuracy']:.4f}")
            print(f"     F1-Score: {scores['f1_score']:.4f}")

print("\n" + "=" * 70)
