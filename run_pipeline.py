"""Train and evaluate sentiment analysis models on customer reviews."""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Silence joblib physical core warning on Windows

import json
import zipfile
from pathlib import Path

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils import clean_text, label_sentiment, predict_with_rules

# Directory setup
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def extract_reviews() -> pd.DataFrame:
    """Extract reviews from ZIP files, clean, and label sentiment."""
    print("Extracting reviews from ZIP files...")
    
    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No ZIP files found in data/ directory")
    
    reviews = []
    
    for zip_path in zip_files:
        print(f"  • {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for filename in zf.namelist():
                    if not filename.endswith('.txt') or filename.startswith('__MACOSX'):
                        continue
                    with zf.open(filename) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        for line in content.splitlines():
                            line = line.strip()
                            if len(line) > 5:
                                reviews.append(line)
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"  Total raw reviews read: {len(reviews)}")
    
    # 1. Dataset Diagnostic Validation & Cleaning
    df_raw = pd.DataFrame({"Review": reviews})
    
    raw_count = len(df_raw)
    
    # Remove empty/corrupted/blank rows
    df_raw = df_raw[df_raw["Review"].str.strip().str.len() > 0]
    non_empty_count = len(df_raw)
    corrupted_count = raw_count - non_empty_count
    
    # Detect and prune duplicate reviews
    dup_count = df_raw.duplicated(subset=["Review"]).sum()
    df_raw = df_raw.drop_duplicates(subset=["Review"], keep="first")
    dedup_count = len(df_raw)
    
    # Text cleaning
    df_raw["Cleaned_Review"] = df_raw["Review"].apply(clean_text)
    
    # Remove rows where cleaned text is empty or too short (noise/empty after cleaning)
    df_clean = df_raw[df_raw["Cleaned_Review"].str.split().str.len() >= 2].reset_index(drop=True)
    clean_count = len(df_clean)
    short_removed_count = dedup_count - clean_count
    
    # Lexicon labeling
    df_clean["Sentiment"] = df_clean["Review"].apply(label_sentiment)
    
    # 2. Output Validation Diagnostics to Console
    print("\n🔍 DATASET VALIDATION DIAGNOSTICS:")
    print(f"  • Raw reviews extracted:        {raw_count}")
    print(f"  • Corrupt/Empty records pruned:  {corrupted_count}")
    print(f"  • Duplicate records dropped:    {dup_count}")
    print(f"  • Short/Noisy reviews dropped:  {short_removed_count}")
    print(f"  • High-quality reviews kept:    {clean_count}")
    
    print("\n📊 LABEL CONSISTENCY & CLASS DISTRIBUTION:")
    class_counts = df_clean["Sentiment"].value_counts()
    for sentiment, count in class_counts.items():
        percentage = (count / clean_count) * 100
        print(f"  • Class '{sentiment:8}': {count:5d} samples ({percentage:.2f}%)")
        
    required_classes = {"Positive", "Negative", "Neutral"}
    found_classes = set(df_clean["Sentiment"].unique())
    missing_classes = required_classes - found_classes
    
    if len(missing_classes) == 0:
        print("  ✓ All 3 sentiment classes exist properly in the dataset.")
    else:
        print(f"  ⚠️ Missing classes: {missing_classes}")
        
    print("\n✓ Dataset validation complete!")
    
    # Save processed dataset
    output_path = PROCESSED_DIR / "cleaned_reviews.csv"
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✓ Cleaned reviews saved to {output_path.name}")
    
    # Save directly to Excel as mandated by assignment requirements
    excel_path = DATA_DIR / "telecom_dataset.xlsx"
    df_clean[["Review", "Sentiment"]].rename(columns={"Review": "Sentence", "Sentiment": "Label"}).to_excel(excel_path, index=False)
    print(f"✓ Saved corpus dataset directly to Excel: {excel_path.name}\n")
    
    return df_clean


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balance the classes in the dataframe by downsampling the majority class."""
    print("⚖️ Balancing dataset classes to prevent model bias...")
    
    class_counts = df["Sentiment"].value_counts()
    print("  Original Class Counts:")
    for sentiment, count in class_counts.items():
        print(f"    • {sentiment:10}: {count}")
        
    min_class_size = class_counts.min()
    if min_class_size < 10:
        print("  Warning: Smallest class size is too small. Skipping downsampling.")
        return df
        
    balanced_dfs = []
    for sentiment in df["Sentiment"].unique():
        sentiment_df = df[df["Sentiment"] == sentiment]
        # Downsample to match the minority class count
        downsampled_df = sentiment_df.sample(n=min_class_size, random_state=42)
        balanced_dfs.append(downsampled_df)
        
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  ✓ Balanced dataset: {len(balanced_df)} samples ({min_class_size} per class)\n")
    return balanced_df


def train_unsupervised(df: pd.DataFrame, vectorizer, X):
    """Perform KMeans unsupervised clustering on TF-IDF features and map clusters to ground truth sentiments."""
    print("⚙️ Running Unsupervised KMeans Clustering...")
    
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Fit KMeans with K=3 (Positive, Negative, Neutral)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate cluster distribution
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("  Cluster Distribution Summary:")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"    • Cluster {cluster_id}: {count:5d} samples")
        
    # Map clusters to sentiment labels using mode alignment
    cluster_mapping = {}
    y_true = df["Sentiment"].values
    
    for cluster_id in range(3):
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            cluster_mapping[cluster_id] = "Neutral"
            continue
        labels_in_cluster = y_true[indices]
        majority_sentiment = pd.Series(labels_in_cluster).mode()[0]
        cluster_mapping[cluster_id] = majority_sentiment
        print(f"    • Cluster {cluster_id} mapped to Ground-Truth Sentiment: {majority_sentiment}")
        
    # Save clustering models and outputs
    joblib.dump(kmeans, MODELS_DIR / "kmeans.pkl")
    mapping_path = RESULTS_DIR / "kmeans_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in cluster_mapping.items()}, f, indent=2)
        
    print(f"✓ Unsupervised KMeans models saved successfully")
    return kmeans, cluster_labels, cluster_mapping


def train_models(df: pd.DataFrame):
    """Train three sentiment classifiers and save models with metrics."""
    print("Partitioning train and test sets...")
    
    # 1. Stratified split on raw dataset first to preserve natural distribution in test set
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Sentiment"]
    )
    
    # 2. Balance ONLY the training set to prevent model bias
    train_df_balanced = balance_dataset(train_df)
    
    X_train_raw = train_df_balanced["Cleaned_Review"]
    y_train = train_df_balanced["Sentiment"]
    
    X_test_raw = test_df["Cleaned_Review"]
    y_test = test_df["Sentiment"]
    
    print("Training models...\n")
    
    # 3. Fit TF-IDF on balanced training set, and transform test set
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # Save vectorizer
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✓ Vectorizer saved")
    print(f"✓ Data split: {X_train.shape[0]} balanced train samples, {X_test.shape[0]} natural test samples")
    
    # 🔍 DEBUGGING: Feature Matrix Shape Checks
    print(f"🔍 DEBUG: Feature Matrix Shape Diagnostics:")
    print(f"  • X_train shape: {X_train.shape} (Samples, Features)")
    print(f"  • X_test shape:  {X_test.shape} (Samples, Features)\n")
    
    # 4. Initialize classifiers
    models = {
        "logistic": LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight="balanced",
            C=0.8  # Slight regularization to prevent overfitting on noisy text
        ),
        "naive_bayes": MultinomialNB(alpha=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight="balanced",
            max_depth=25,          # Regulate tree depth to prevent extreme overfitting on rare words
            min_samples_split=5    # Prevent overfitting on individual reviews
        )
    }
    
    metrics = {}
    
    filename_map = {
        "logistic": "logistic_metrics.pkl",
        "naive_bayes": "nb_metrics.pkl",
        "random_forest": "rf_metrics.pkl"
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Compute each metric independently using correct sklearn functions
        accuracy = accuracy_score(y_test, predictions)
        
        # Weighted averages (UI compatibility)
        precision_weighted = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Macro averages (Richer quality measurement)
        precision_macro = precision_score(y_test, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=["Positive", "Neutral", "Negative"])
        
        metrics[name] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision_weighted), 4),
            "precision_macro": round(float(precision_macro), 4),
            "recall": round(float(recall_weighted), 4),
            "recall_macro": round(float(recall_macro), 4),
            "f1_score": round(float(f1_weighted), 4),
            "f1_macro": round(float(f1_macro), 4)
        }
        
        # Save model pkl
        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        
        # Save structured model-specific metrics .pkl file
        model_metrics = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision_weighted), 4),
            "recall": round(float(recall_weighted), 4),
            "f1_score": round(float(f1_weighted), 4),
            "confusion_matrix": cm
        }
        metrics_pkl_path = RESULTS_DIR / filename_map[name]
        joblib.dump(model_metrics, metrics_pkl_path)
        
        print(f"======================================================================")
        print(f"🤖 MODEL: {name.upper()}")
        print(f"======================================================================")
        print(f"  Accuracy:         {accuracy:.4f}")
        print(f"  Precision (W):    {precision_weighted:.4f}  |  Precision (Macro): {precision_macro:.4f}")
        print(f"  Recall (W):       {recall_weighted:.4f}  |  Recall (Macro):    {recall_macro:.4f}")
        print(f"  F1-Score (W):     {f1_weighted:.4f}  |  F1-Score (Macro):  {f1_macro:.4f}")
        
        # 🔍 DEBUGGING: Prediction Distribution Summary
        pred_counts = pd.Series(predictions).value_counts()
        dist_str = ", ".join([f"{k}: {v}" for k, v in pred_counts.items()])
        print(f"  🔍 DEBUG: Prediction Distribution: {dist_str}\n")
        
        # Print classification report
        print(f"  Classification Report for {name.upper()}:")
        print(classification_report(y_test, predictions, zero_division=0))
        
        # Print Confusion Matrix
        print(f"  Confusion Matrix for {name.upper()}:")
        print("               Pred: Pos   Neu   Neg")
        print(f"   Actual: Pos  {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}")
        print(f"           Neu  {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}")
        print(f"           Neg  {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}\n")
    
    # Save metrics
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Models and metrics saved successfully")
    
    # 5. Unsupervised KMeans training on all TF-IDF vectors
    X_all = vectorizer.transform(df["Cleaned_Review"].fillna(""))
    train_unsupervised(df, vectorizer, X_all)
    
    # Prediction validation tests including Mixed/Contrast handling edge cases
    print("\n🔬 Running Prediction Validation Tests:")
    validation_cases = {
        "Positive": ["excellent internet service", "very fast network"],
        "Negative": ["worst telecom experience", "slow and bad connection"],
        "Neutral": ["service is average", "network is okay"],
        "Mixed/Contrast": [
            "the customer service was bad but the internet speed is excellent",
            "the signal was great however the price is worst and expensive"
        ]
    }
    
    for actual_sent, sentences in validation_cases.items():
        print(f"\n  • Expected {actual_sent.upper()}:")
        for text in sentences:
            lr_out = predict_with_rules(text, vectorizer, models["logistic"])
            nb_out = predict_with_rules(text, vectorizer, models["naive_bayes"])
            rf_out = predict_with_rules(text, vectorizer, models["random_forest"])
            
            print(f"    Review: \"{text}\"")
            print(f"      - Logistic:     {lr_out}")
            print(f"      - Naive Bayes:  {nb_out}")
            print(f"      - Random Forest:{rf_out}")


if __name__ == "__main__":
    try:
        df = extract_reviews()
        train_models(df)
        print("\n✓ Pipeline complete! Run UI with: streamlit run ui/app.py")
    except Exception as e:
        print(f"\nError: {e}")
