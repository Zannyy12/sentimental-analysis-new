# 📶 Telecom Sentiment Analysis: Production-Grade ML Pipeline & Interactive Dashboard

**A Comprehensive Data Science & Machine Learning Academic Report**  
*Course: Introduction to Data Science (6th Semester Project)*  

---## 📖 Executive Summary & Project Overview

This project implements a complete, academic-grade, and demo-ready Sentiment Analysis system designed for Telecom customer reviews. It integrates **Supervised Machine Learning** with **Unsupervised Text Clustering** inside an interactive analytical pipeline:
1. **Data Pipeline & Excel Dataset Generation**: Reads raw `.txt` corpus files line-by-line, runs an NLP negation-preserving preprocessing layer, applies rule-based contextual sentiment labeling (`Positive`, `Negative`, `Neutral`), and compiles them into a clean, mandated Excel dataset (`data/telecom_dataset.xlsx`) with standard columns: `Sentence | Label`.
2. **Interactive Streamlit Dashboard**: Provides a high-fidelity interface with live uploading (CSV/XLSX), brand/provider dropdown category filtering, live action triggers (live training, live clustering, live benchmarks plotting), and real-time inference prediction.
3. **Supervised ML Classifiers**: Performs a Stratified 80/20 train/test split, TF-IDF vectorization, class balancing downsampling, and trains three distinct classifiers: Logistic Regression, Multinomial Naive Bayes, and Random Forest.
4. **Unsupervised K-Means Text Clustering (MANDATORY)**: Trains a K-Means model ($K=3$) using TF-IDF features on the active dataset, determines cluster size distributions, automatically aligns/maps clusters to positive, negative, and neutral labels using statistical mode alignment, and projects clusters into 2D space using Principal Component Analysis (PCA) for visual exploration.

---

## 📂 Project Architecture

The repository is structured strictly with a focus on modularity, zero external database dependencies, and complete compliance with academic assignment guidelines:

```
├── data/                       # Raw input data and Excel deliverables
│   ├── telecom_dataset.xlsx    # Mandated raw/labeled Excel dataset (Sentence | Label)
│   └── *.zip                   # Original input compressed review archives
├── processed/                  # Cleaned and processed dataset
│   └── cleaned_reviews.csv     # Cleaned reviews with final rule-based labels
├── models/                     # Serialized binary artifacts
│   ├── vectorizer.pkl          # Saved fitted TF-IDF Vectorizer
│   ├── logistic.pkl            # Trained Logistic Regression model
│   ├── naive_bayes.pkl         # Trained Multinomial Naive Bayes model
│   ├── random_forest.pkl       # Trained Random Forest Classifier
│   └── kmeans.pkl              # Trained KMeans Unsupervised model
├── results/                    # Validation metrics and benchmarks
│   ├── metrics.json            # Model performance records (F1, Accuracy, etc.)
│   └── kmeans_mapping.json     # Cluster ID to Sentiment Label alignments
├── ui/                         # Presentation Layer
│   └── app.py                  # Streamlit web dashboard (uploads, filters, triggers)
├── utils.py                    # Shared core utilities (cleaning & labeling)
├── run_pipeline.py             # Pipeline master script (CLI dataset generation & training)
├── requirements.txt            # Minimal dependency list (adds openpyxl for Excel)
└── README.md                   # Academic project report (this document)
```

---

## ⚙️ Data Pipeline & Quality Engineering (`run_pipeline.py` & `utils.py`)

### 1. Data Cleaning & The "Negation Preservation" NLP Technique
Standard NLP libraries like NLTK or SpaCy use broad stopword lists that remove negation words (such as *not, don't, doesn't, can't, isn't*). In sentiment analysis, removing these words completely destroys the context:
* *Standard Cleaning*: `"network is not good"` $\rightarrow$ `['network', 'good']` (Incorrectly classified as **Positive**)
* *Our Preserving Cleaning*: `"network is not good"` $\rightarrow$ `['network', 'not', 'good']` (Correctly classified as **Negative**)

To resolve this, we implemented **Contraction Standardization** and **Minimal Stopword Selection**:
* **Contraction Standardizer**: Before punctuation removal, contractions ending in `n't` are standardized using regular expressions (e.g., `don't` $\rightarrow$ `dont`, `doesn't` $\rightarrow$ `doesnt`).
* **Negation Shielding**: Negations (e.g., `not, dont, doesnt, cant, without, never`) are systematically excluded from the custom `STOPWORDS` list.
* **Corrupt & Duplicate Filtering**: Duplicate entries and short reviews containing fewer than 2 words are automatically filtered out during dataset initialization.

```python
# Regex to preserve and standardise negation terms before punctuation stripping
text = re.sub(r"\b(\w+)n't\b", r"\1nt", text)
```

### 2. Contextual Weighted Lexicon Tagging (Weak Supervision)
To label the raw corpus, the system implements an advanced **Weighted Lexicon** tagging algorithm with contextual negation checking and neutral damping:
* **Strong Positive (Weight 2.0)**: *excellent, amazing, wonderful, fantastic, awesome, love, perfect, superb, brilliant, outstanding, speedy, resolved, solved*
* **Moderate Positive (Weight 1.0)**: *good, great, nice, satisfied, satisfy, satisfactory, fast, best, happy, recommend, helpful, thank, thanks, clear, properly, strong, impressive, pleased, fine, working, decent, super*
* **Strong Negative (Weight 2.0)**: *worst, pathetic, terrible, awful, horrible, hate, useless, rubbish, nonsense, disconnect, broken, unreliable*
* **Moderate Negative (Weight 1.0)**: *bad, poor, slow, fail, failed, failure, dropped, drop, expensive, charge, charges, billing, issue, disappointed, unhappy, down, weak, costly, annoying, frustrating, disappointing, waste, unfair, problem, delay, late, slowly*
* **Neutral Words**: *average, okay, ok, normal, standard, moderate, ordinary, fairly, satisfactory*

#### The Algorithm & Damping:
1. **Preceding 3-Word Negation Window**: When a keyword is matched, our algorithm scans 3 words backwards for negation terms (e.g., `not`, `never`, `dont`). If found, it flips polarity (e.g., positive becomes negative, and negative becomes mildly positive/neutral).
2. **Neutral Damping**: If any neutral word (e.g., *average*, *okay*, *ok*) is detected, and the margin between Positive and Negative scores is narrow ($\le 1.0$), the review is automatically assigned **Neutral**. This prevents borderline reviews from skewing validation results.

### 3. Automated Class Balancing (Downsampling)
Rule-based taggers naturally produce a high volume of **Neutral** records, causing dataset imbalance. Training models on skewed data biases them to always predict the majority class. 
Our pipeline automatically detects imbalance and performs **under-sampling on the majority classes**, downsampling them to match the exact size of the minority class. 

---

## 📈 Feature Engineering & Supervised ML Classifiers

### 1. TF-IDF Vectorization
The text is mapped into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**:
* **Unigrams & Bigrams**: (`ngram_range=(1, 2)`) captures single words and two-word pairings (e.g., *"excellent speed"*, *"slow internet"*).
* **Vocabulary Quality Control**: `min_df=2` automatically drops spelling mistakes and extremely rare words, while `max_features=5000` prevents high dimensionality.
* **Fitted Vectorizer Reuse**: The vectorizer fitted on the training set is serialized as `vectorizer.pkl` and reused in the UI for inference to avoid feature mismatch.

### 2. Trained Classifiers & Regularization
We train three distinct models, adding L2 and decision-tree regularization parameters to prevent overfitting:
1. **Logistic Regression (Baseline Linear Model)**: 
   - Hyperparameters: `max_iter=1000`, `class_weight="balanced"`, `C=0.8`.
   - The slightly smaller inverse regularization strength `C` prevents overfitting on rare terms.
2. **Multinomial Naive Bayes (Probabilistic Classifier)**:
   - Hyperparameters: `alpha=1.0` (Laplace smoothing).
   - High performance on sparse term-frequency vectors.
3. **Random Forest Classifier (Ensemble Model)**:
   - Hyperparameters: `n_estimators=100`, `class_weight="balanced"`, `max_depth=25`, `min_samples_split=5`, `random_state=42`.
   - Limiting `max_depth` and raising `min_samples_split` prevents the ensemble from overfitting on rare word patterns, providing extremely robust out-of-sample generalizations.

### 3. Rule-Based Inference Layer (Mixed Sentiment Handling)
While ML models perform well, they occasionally struggle with complex mixed sentiment logic. At inference time, the predictions run through a strict **Rule-Based Pre-check Layer** before hitting the ML models:
* **Exact Phrase Overrides**: Immediately captures terms like *"worst experience"* or *"fast internet"*.
* **Contrast Connector Parsing**: When users write compound reviews using *'but'* or *'however'* (e.g., *"Service is bad but speed is great"*), the layer isolates the final clause, effectively resolving complex mixed-sentiment statements by prioritizing the concluding sentiment.


---

## 📊 Scientific Evaluation & Validation Results

### 1. Stratified Natural Validation Set
To ensure our evaluation is mathematically trustworthy and matches real-world scenarios, the project implements a professional **Natural Validation Splitting** strategy:
* **Natural Test Set**: We perform a Stratified 80/20 train/test split on the original corpus *first*. This preserves the real-world, natural class distribution of customer reviews in the test set.
* **Balanced Training Set**: We downsample *only* the training set (`train_df`) to match class sizes, preventing learning bias.
* **Varying, Trustworthy Metrics**: Because the models are evaluated on a natural validation set, Accuracy, Precision, Recall, and F1-score differ naturally and accurately reflect actual real-world performance instead of artificial mathematical equivalence.

### 2. Rich Debugging & Telemetry Layer
To ensure simple and robust maintenance, the pipeline outputs clear terminal telemetries:
* **Dataset Diagnostics**: Counts and reports corrupt rows, duplicates, and short noise removed.
* **Feature Shape Diagnostics**: Performs shape checks on `X_train` and `X_test` matrices (e.g. `X_train shape: (samples, features)`).
* **Prediction Distribution Summaries**: Computes the exact counts of Positive/Negative/Neutral labels predicted by each model on the test partition to actively guard against prediction collapse.

### 3. Metric Calculations (Weighted + Macro)
For each model, the pipeline computes and prints:
* **Weighted Averages**: Overall Accuracy, Precision, Recall, and F1-score, weighted by class representation (perfectly aligned with Streamlit progress indicators).
* **Macro Averages**: Unweighted averages of precision, recall, and F1-score computed per class (ideal for deep scientific assessment of minority classes).
* **Confusion Matrices**: Shows precise class separations.
* **Classification Reports**: Fully detailed standard performance tables.

### 4. Robust Real Prediction Test Suite
The pipeline validates model outputs against real-world and complex edge cases:
* **Positive Validation**: *"excellent internet service"*, *"very fast network"* $\rightarrow$ Predicts **Positive**.
* **Negative Validation**: *"worst telecom experience"*, *"slow and bad connection"* $\rightarrow$ Predicts **Negative**.
* **Neutral Validation**: *"service is average"*, *"network is okay"* $\rightarrow$ Predicts **Neutral**.
* **Mixed/Contrast Validation**: 
  - *"the customer service was bad but the internet speed is excellent"* $\rightarrow$ Correctly boosted and classified as **Positive**.
  - *"the signal was great however the price is worst and expensive"* $\rightarrow$ Correctly boosted and classified as **Negative**.

---

## 🖥️ Sleek, High-Fidelity UI Presentation (`ui/app.py`)

The user interface is designed to emulate premium SaaS monitoring telemetry tools:
* **Premium Glassmorphic Design**: Built using modern, cohesive, dark-themed styling (`#0b0e14` page body, `#11141e` container cards, `#5b4ef6` high-contrast buttons).
* **Parallel Prediction Telemetry**: Allows entering reviews and instantly displays side-by-side predictions for all 3 models.
* **Animated Benchmark Progress Bars**: Displays Accuracy, Precision, and Recall values as animated, color-coded progress bars (using a blue/green/yellow tailored palette).
* **Dynamic Confusion Matrices**: The UI provides interactive toggles for Confusion Matrices, allowing users to switch between a visual **Heatmap View** (using seaborn styling) and a raw **Table View**.
* **Graceful Fallback Mode**: If `vectorizer.pkl` is missing, the dashboard runs in high-fidelity template fallback mode with a helpful warning message instructing the user how to generate the model files.

---

## 🚀 Quick Start Guide

### 1. Set Up Environment
Install required scientific packages:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Extract ZIP files, balance classes, train models, save artifacts, and run tests:
```bash
python run_pipeline.py
```

### 3. Launch Dashboard
Start the interactive Streamlit telemetry dashboard:
```bash
python -m streamlit run ui/app.py
```

---

## 🎓 Academic Defense Q&A (Preparing for Your Viva)

Be prepared to answer these common questions during your project defense:

1. **Why split the pipeline from the UI app?**
   * *Answer*: "In industry deployments, training is an offline, resource-heavy process. Deploying a model to serve user predictions (inference) should be near-instantaneous. By saving the trained models to `.pkl` files, our Streamlit UI launches instantly and runs inference in milliseconds without needing to rebuild or retrain on the raw data."
2. **What is TF-IDF and why did you choose it over simple word counting?**
   * *Answer*: "Simple word counting (Bag of Words) counts terms equally. TF-IDF penalizes extremely frequent words like 'network' or 'telecom' that appear in every review and provides a higher mathematical score to distinctive sentiment terms like 'worst', 'pathetic', or 'excellent'."
3. **How does your code prevent predictions from being biased toward a single class?**
   * *Answer*: "We implemented an automated under-sampling function that downsamples the majority sentiment classes to match the minority class. Combined with stratified test splitting and `class_weight='balanced'` in our classifiers, we protect our boundary vectors from majority class dominance."
4. **How do you handle negation like 'not good' or 'cant use'?**
   * *Answer*: "We developed a regular expression contraction standardizer that maps terms like `n't` to `nt` before removing punctuation. We then excluded negations from our `STOPWORDS` list and utilized a 3-word preceding search window in our labeling lexicon to invert sentiment polarities."
5. **How does your pipeline handle mixed sentiment like 'bad signal but fast internet'?**
   * *Answer*: "We implemented a custom inference rule layer that acts before the ML prediction. It splits the sentence at contrast connectors like 'but' or 'however', isolates the second half, scans for sentiment terms, and prioritizes the concluding sentiment to provide the final prediction."
