"""Shared utilities for text preprocessing and model inference."""

import re

# Stopwords - CRITICAL: removed negation words and single-letter contractions that affect sentiment
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 's', 't', 'can', 'will', 
    'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}

# Sentiment keywords for rule-based labeling
# Module-level Lexicon definitions
STRONG_POSITIVE = {
    'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'love', 'perfect', 
    'superb', 'brilliant', 'outstanding', 'speedy', 'resolved', 'solved'
}
MODERATE_POSITIVE = {
    'good', 'great', 'nice', 'satisfied', 'satisfy', 'satisfactory', 'fast', 'best', 
    'happy', 'recommend', 'helpful', 'thank', 'thanks', 'clear', 'properly', 'strong', 
    'impressive', 'pleased', 'fine', 'working', 'decent', 'super'
}

STRONG_NEGATIVE = {
    'worst', 'pathetic', 'terrible', 'awful', 'horrible', 'hate', 'useless', 
    'rubbish', 'nonsense', 'disconnect', 'broken', 'unreliable'
}
MODERATE_NEGATIVE = {
    'bad', 'poor', 'slow', 'fail', 'failed', 'failure', 'dropped', 'drop', 
    'expensive', 'charge', 'charges', 'billing', 'issue', 'disappointed', 
    'unhappy', 'down', 'weak', 'costly', 'annoying', 'frustrating', 
    'disappointing', 'waste', 'unfair', 'problem', 'delay', 'late', 'slowly'
}

NEUTRAL_WORDS = {
    'average', 'okay', 'ok', 'normal', 'standard', 'moderate', 'ordinary', 
    'fairly', 'satisfactory'
}

NEGATION_TERMS = {
    'not', 'no', 'never', 'none', 'neither', 'nor', 'dont', 'doesnt', 'didnt', 'cant', 
    'cannot', 'isnt', 'arent', 'wasnt', 'weren', 'werent', 'havent', 'hasnt', 'hadnt', 
    'wont', 'wouldnt', 'shouldnt', 'couldnt', 'without'
}

# Keep original flat lists for backward compatibility if imported elsewhere
POSITIVE_KEYWORDS = STRONG_POSITIVE.union(MODERATE_POSITIVE)
NEGATIVE_KEYWORDS = STRONG_NEGATIVE.union(MODERATE_NEGATIVE)


def clean_text(text: str) -> str:
    """Clean and preprocess text: lowercase, preserve contractions, remove punctuation, and remove stopwords."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    
    # Handle common contractions to preserve negation before punctuation removal
    text = re.sub(r"\b(\w+)n't\b", r"\1nt", text)
    text = text.replace("can't", "cant")
    text = text.replace("won't", "wont")
    text = text.replace("cannot", "cannot")
    
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Keep only letters and spaces
    text = " ".join(text.split())  # Remove extra whitespace
    
    # Remove stopwords
    words = text.split()
    cleaned = [w for w in words if w not in STOPWORDS]
    
    return " ".join(cleaned)


def label_sentiment(text: str) -> str:
    """Label sentiment using weighted keyword matching with preceding-window negation and contrast weighting."""
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
        
    text_lower = text.lower()
    # Tokenize words (lowercase alphabetical tokens with apostrophes)
    words = re.findall(r"[a-z']+", text_lower)
    
    if not words:
        return 'Neutral'
        
    pos_score = 0.0
    neg_score = 0.0
    neutral_count = 0
    
    # Pre-process contractions in the word stream
    cleaned_words = []
    for w in words:
        if w.endswith("n't"):
            cleaned_words.append(w[:-3] + "nt")
        elif w == "can't":
            cleaned_words.append("cant")
        elif w == "won't":
            cleaned_words.append("wont")
        else:
            cleaned_words.append(w)
            
    # Contrast Weighting Strategy:
    # Reviews containing contrast connectors ('but', 'however') shift the primary focus to the final clause.
    # Therefore, terms located after a contrast connector receive an extra weight multiplier (e.g. 1.8x).
    current_multiplier = 1.0
    
    for i, word in enumerate(cleaned_words):
        if word in {'but', 'however'}:
            current_multiplier = 1.8  # Boost concluding sentiment in contrast statements
            continue
            
        is_strong_pos = word in STRONG_POSITIVE or any(word.startswith(kw) for kw in STRONG_POSITIVE if len(kw) > 3)
        is_mod_pos = word in MODERATE_POSITIVE or any(word.startswith(kw) for kw in MODERATE_POSITIVE if len(kw) > 3)
        
        is_strong_neg = word in STRONG_NEGATIVE or any(word.startswith(kw) for kw in STRONG_NEGATIVE if len(kw) > 3)
        is_mod_neg = word in MODERATE_NEGATIVE or any(word.startswith(kw) for kw in MODERATE_NEGATIVE if len(kw) > 3)
        
        is_neutral = word in NEUTRAL_WORDS
        
        if is_neutral:
            neutral_count += 1
            
        is_pos = is_strong_pos or is_mod_pos
        is_neg = is_strong_neg or is_mod_neg
        
        if is_pos or is_neg:
            # Check negation in the preceding window of 3 words
            negated = False
            start_idx = max(0, i - 3)
            for j in range(start_idx, i):
                if cleaned_words[j] in NEGATION_TERMS:
                    negated = True
                    break
            
            # Base weight
            base_weight = 2.0 if (is_strong_pos or is_strong_neg) else 1.0
            # Apply contrast clause multiplier
            effective_weight = base_weight * current_multiplier
            
            if negated:
                if is_pos:
                    neg_score += effective_weight
                if is_neg:
                    pos_score += (effective_weight * 0.5)
            else:
                if is_pos:
                    pos_score += effective_weight
                if is_neg:
                    neg_score += effective_weight
                    
    # Decision boundaries
    if pos_score > neg_score:
        diff = pos_score - neg_score
        # If neutral word is present and the positive sentiment is weak/moderate, damp to Neutral
        if neutral_count > 0 and diff <= 1.0:
            return 'Neutral'
        return 'Positive'
    elif neg_score > pos_score:
        diff = neg_score - pos_score
        # If neutral word is present and the negative sentiment is weak/moderate, damp to Neutral
        if neutral_count > 0 and diff <= 1.0:
            return 'Neutral'
        return 'Negative'
    else:
        return 'Neutral'


def predict_with_rules(text: str, vectorizer, model) -> str:
    """Pre-check rules before using the model for prediction, handling negation and contrast."""
    if not isinstance(text, str) or not text.strip():
        return "Neutral"
        
    text_lower = text.lower()
    
    # 1. Specific exact phrase matches for strong negatives & positives
    neg_phrases = [
        "not good", "no good", "dont like", "dislike", "not satisfied", 
        "very bad", "worst experience", "bad connection", "slow speed", 
        "drops frequently", "frustrating"
    ]
    pos_phrases = [
        "not bad", "not slow", "not worst", "excellent network", 
        "very good", "great speed", "fast internet"
    ]
    
    for phrase in neg_phrases:
        if phrase in text_lower:
            return "Negative"
            
    for phrase in pos_phrases:
        if phrase in text_lower:
            return "Positive"
            
    # 2. Contrast connector handling: focus on second half if it contains 'but' or 'however'
    if 'but' in text_lower or 'however' in text_lower:
        parts = re.split(r'\b(?:but|however)\b', text_lower, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            second_half = parts[1].strip()
            # Clean the second half
            second_cleaned = clean_text(second_half)
            
            # Words to scan for sentiment in second half
            pos_words = ["good", "great", "excellent", "fast", "satisfied", "satisfy", "happy", "best", "love", "amazing"]
            neg_words = ["bad", "poor", "slow", "expensive", "worst", "pathetic", "terrible", "billing", "charges", "annoying", "frustrating", "costly"]
            
            has_pos = any(w in second_cleaned for w in pos_words)
            has_neg = any(w in second_cleaned for w in neg_words)
            has_negation = any(neg in second_half for neg in ["not", "dont", "doesnt", "cant", "no"])
            
            if has_pos and not has_negation and not has_neg:
                return "Positive"
            if has_neg and not has_negation and not has_pos:
                return "Negative"
                
    # 3. Fallback to standard ML model prediction
    cleaned = clean_text(text)
    x_vec = vectorizer.transform([cleaned])
    return model.predict(x_vec)[0]
