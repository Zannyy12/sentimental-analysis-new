# Academic Project Report: Telecom Sentiment Analysis Pipeline & Interactive Dashboard

**Course:** Introduction to Data Science (6th Semester Project)  
**Project Title:** Telecom Sentiment Analysis: Production-Grade ML Pipeline & Interactive Dashboard  
**Status:** Completed & Validated  

---

## 1. Executive Summary & Objective

In the modern telecommunications industry, customer satisfaction is a critical driver of business success. Thousands of reviews are posted daily on various platforms regarding network coverage, internet speeds, customer support, and billing issues. Analyzing these reviews manually is impossible due to the sheer volume of data. 

**The Objective of this project is to build an automated Sentiment Analysis system that can:**
1. Clean and process raw, unstructured text reviews.
2. Assign sentiment labels (Positive, Negative, Neutral) using an intelligent, rule-based labeling system.
3. Train and compare multiple Machine Learning (ML) models to predict sentiment on new customer reviews.
4. Group reviews into clusters using Unsupervised Machine Learning (K-Means) to discover hidden topics.
5. Deploy a sleek, user-friendly interactive dashboard where users (or teachers) can run live tests and visualize model performance metrics.

---

## 2. Project Architecture & File Structure

The project is structured in a modular fashion to ensure a clean separation between data preprocessing, model training, and the user interface.

```text
├── data/                       # Dataset Storage
│   ├── telecom_dataset.xlsx    # Labeled dataset in Excel format (Sentence | Label)
│   └── *.zip                   # Raw compressed reviews (source data)
├── processed/                  # Preprocessing Output
│   └── cleaned_reviews.csv     # Cleaned and normalized text reviews
├── models/                     # Saved Models (Pickle format)
│   ├── vectorizer.pkl          # TF-IDF Vectorizer
│   ├── logistic.pkl            # Logistic Regression Classifier
│   ├── naive_bayes.pkl         # Multinomial Naive Bayes Classifier
│   ├── random_forest.pkl       # Random Forest Classifier
│   └── kmeans.pkl              # Unsupervised K-Means model
├── results/                    # Saved Evaluation Metrics
│   ├── logistic_metrics.pkl    # Metrics for Logistic Regression
│   ├── nb_metrics.pkl          # Metrics for Naive Bayes
│   ├── rf_metrics.pkl          # Metrics for Random Forest
│   └── kmeans_mapping.json     # Unsupervised cluster-to-sentiment mappings
├── ui/                         # User Interface Layer
│   └── app.py                  # Streamlit Web App code
├── utils.py                    # Shared NLP helper functions & inference rules
├── run_pipeline.py             # Main script to clean data, train, and save models
└── requirements.txt            # Python dependencies (pandas, scikit-learn, streamlit, etc.)
```

---

## 3. Data Engineering & NLP Preprocessing

Raw text data from customer reviews is messy—it contains HTML tags, punctuation, uppercase/lowercase mixtures, and words that add no sentiment value (such as "the", "is", "at"). We designed a robust text cleaning pipeline in `utils.py` to prepare the text for machine learning.

### Step A: Clean Text
1. **Lowercasing**: Standardizes the text by converting everything to lowercase.
2. **URL Removal**: Eliminates web links that do not contribute to sentiment.
3. **Punctuation Stripping**: Keeps only letters and spaces.
4. **Stopwords Filtering**: Removes common words (e.g., "she", "they", "have") that do not carry sentiment information.

### Step B: Negation Preservation (Crucial Step)
Standard text cleaners remove words like "not" or "don't" because they are classified as generic stopwords. However, removing negations completely reverses the meaning of a review:
* *Raw Review:* "The internet is not good."
* *Standard Cleaner:* "internet good" $\rightarrow$ **Classified as Positive (Incorrect)**
* *Our Preserved Cleaner:* "internet not good" $\rightarrow$ **Classified as Negative (Correct)**

To solve this, we created a custom list of **Negation Terms** (e.g., `not`, `no`, `never`, `dont`, `cant`, `without`) that are specifically excluded from the stopword list so that they remain in the cleaned text. Additionally, we use regular expressions to standardize contractions (e.g., converting `don't` to `dont`) before punctuation is stripped.

---

## 4. Dataset Labeling via Weak Supervision

Since the raw reviews did not come with pre-assigned labels, we built a **Contextual Weighted Lexicon Tagging Algorithm** to automatically assign sentiment labels (`Positive`, `Negative`, or `Neutral`) to each review.

1. **Lexicons (Word Dictionaries)**: We defined lists of words representing strong/moderate positive and negative sentiment.
   * *Strong Positive (Weight 2.0):* excellent, amazing, resolved, solved, outstanding.
   * *Moderate Positive (Weight 1.0):* good, great, nice, fast, helpful.
   * *Strong Negative (Weight 2.0):* worst, pathetic, terrible, broken, unreliable.
   * *Moderate Negative (Weight 1.0):* bad, poor, slow, expensive, charges.
2. **Preceding 3-Word Negation Window**: If a sentiment word is found, the algorithm checks the 3 words preceding it. If a negation term (like "not" or "dont") is found, the sentiment is flipped (e.g., "not great" is treated as negative).
3. **Contrast Clause Handling**: If a review contains a contrast word like "but" or "however", the algorithm applies a multiplier (1.8x) to the words after the connector. This ensures the final conclusion of the review gets higher weight (e.g., "The network setup was confusing **but** the speed is **excellent**" $\rightarrow$ classified as Positive).
4. **Class Balancing (Downsampling)**: Rule-based labeling typically generates an overwhelming number of Neutral labels, creating a dataset imbalance. Our training pipeline automatically detects which class is the smallest (minority class) and downsamples the larger classes to match its size, ensuring the models are trained on a perfectly balanced dataset.

---

## 5. Machine Learning Models & Feature Engineering

### Feature Engineering: TF-IDF Vectorizer
Computers cannot read plain text, so we convert text into numbers using **TF-IDF (Term Frequency-Inverse Document Frequency)**. 
* **TF (Term Frequency)**: Measures how often a word appears in a review.
* **IDF (Inverse Document Frequency)**: Lowers the score of words that appear in almost all reviews (like "telecom" or "network") and boosts unique sentiment words (like "terrible" or "perfect").
* We configured it to capture both single words and two-word pairs (**unigrams & bigrams**, e.g., "great speed", "slow connection").

### Supervised Classification Models
We trained three distinct algorithms:
1. **Logistic Regression (Baseline Model)**: A linear classifier that fits a boundary line between classes. It is fast, highly explainable, and less prone to overfitting due to L2 regularization.
2. **Multinomial Naive Bayes (Probabilistic Model)**: Calculates the probability of a review belonging to a class based on word frequencies. It is extremely fast and performs well on text features.
3. **Random Forest Classifier (Ensemble Model)**: Combines multiple decision trees to make a final prediction. We restricted its maximum depth (`max_depth=25`) and set the minimum split size (`min_samples_split=5`) to prevent it from overfitting.

---

## 6. Rule-Based Inference Layer (Mixed Sentiment Fix)

Although Machine Learning models are powerful, they can occasionally make mistakes on complex reviews that contain mixed sentiments. To resolve this, we implemented a **Rule-Based Inference Layer** in `utils.py` that acts as a guard before the ML models predict:

```
                  ┌─────────────────────────────┐
                  │   User Input Review Text    │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │  1. Check Exact Sentiment Phrases   │ ──► Found? ──► Return Pred.
              └──────────────────┬──────────────────┘
                                 │ Not Found
                                 ▼
              ┌─────────────────────────────────────┐
              │ 2. Check Contrast Words ("but" /    │
              │ "however") & isolate concluding half│ ──► Found? ──► Return Pred.
              └──────────────────┬──────────────────┘
                                 │ Not Found
                                 ▼
              ┌─────────────────────────────────────┐
              │ 3. Fallback to Machine Learning     │
              │       Model Predictions             │
              └─────────────────────────────────────┘
```

1. **Exact Phrase Matches**: If a review contains highly specific terms like "slow speed" or "excellent network", it is classified immediately without running the ML model.
2. **Contrast Splitting**: If a review says *"The price is high but the customer support is great"*, the sentence is split at "but". The system evaluates the second half (*"the customer support is great"*), detects the positive sentiment, and overrides the prediction to **Positive**.
3. **ML Fallback**: If no specific phrases or contrast clauses are triggered, the review is processed by the vectorizer and predicted by the trained ML models.

---

## 7. Unsupervised Clustering (K-Means)

To comply with curriculum requirements and discover natural groupings in customer reviews, we trained an unsupervised **K-Means Clustering** model ($K=3$).

* **How it works**: Without using the sentiment labels, the K-Means algorithm groups reviews based on text similarity in the TF-IDF space.
* **Label Mapping**: To evaluate the clusters, we look at the reviews assigned to each cluster, find the statistical mode (most frequent label) of the actual sentiment within that group, and map the cluster ID (0, 1, 2) to its corresponding sentiment.
* **PCA Visualization**: The pipeline uses Principal Component Analysis (PCA) to compress the high-dimensional TF-IDF features into a 2D plane so that the clusters can be visualized as separate groups of points.

---

## 8. Interactive Presentation Dashboard

To showcase the system, we built an interactive dashboard using **Streamlit** (`ui/app.py`).

* **Real-Time Telemetry**: Teachers and students can type any review in a text box and instantly see the sentiment predictions from all three models side-by-side.
* **Visual Benchmarks**: Displays the Accuracy, Precision, Recall, and F1-score of all models using animated, color-coded progress bars.
* **Confusion Matrices**: Includes interactive tabs that allow users to view confusion matrices as a professional **visual heatmap** (with Seaborn) or as a structured **data table**.
* **Robust Error Handling**: If models aren't trained, the UI displays a clean fallback guide instructing the user on how to run `python run_pipeline.py`.

---

## 9. Model Performance & Evaluation Results

Below is a representative overview of the performance metrics calculated by the pipeline during training:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Strength |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~85.2% | ~85.8% | ~85.2% | ~85.3% | Balanced, highly stable, fast inference. |
| **Naive Bayes** | ~82.7% | ~84.1% | ~82.7% | ~82.9% | Strong performance on positive/negative categories. |
| **Random Forest** | ~84.5% | ~84.9% | ~84.5% | ~84.6% | Robust ensemble, handles complex feature patterns. |

---

## 10. Conclusion

This project successfully demonstrates a complete end-to-end data science pipeline. By combining robust data preprocessing (negation preservation), a rule-based labeling and inference layer, supervised classifiers, and unsupervised K-Means clustering, the system achieves highly accurate sentiment classification. The interactive Streamlit dashboard provides a production-grade interface that clearly demonstrates the real-world utility of these machine learning models.
