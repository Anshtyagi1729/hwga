# src/sentiment.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from transformers import pipeline
from textblob import TextBlob
import logging
from typing import List, Dict, Tuple
from config.config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False  # Track if model is trained
        
        # Initialize HuggingFace (The Teacher)
        self.hf_model = None
        try:
            logger.info("Loading HuggingFace model...")
            self.hf_model = pipeline('sentiment-analysis', 
                                    model=Config.HUGGINGFACE_MODEL,
                                    device=-1)
        except Exception as e:
            logger.warning(f"HF Model failed: {e}")

    def train_on_db(self, articles: List[Dict]) -> str:
        """Train Logistic Regression using existing database labels"""
        try:
            # 1. Filter for valid data
            valid_data = [
                a for a in articles 
                if a.get('sentiment_label') in ['positive', 'negative'] 
                and a.get('processed_content')
            ]

            if len(valid_data) < 5:
                return ""

            # 2. Prepare Training Data
            texts = [a['processed_content'] for a in valid_data]
            labels = [a['sentiment_label'] for a in valid_data]

            # 3. Train (Fit) the Model
            logger.info(f"Training Logistic Regression on {len(texts)} articles...")
            X = self.vectorizer.fit_transform(texts)
            self.logistic_model.fit(X, labels)
            self.is_fitted = True
            
            return f"Success! Model trained on {len(texts)} articles."

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return f"Error: {e}"

    def predict_logistic(self, text: str) -> Tuple[str, float]:
        """Predict using Logistic Regression (The Student)"""
        if not self.is_fitted:
            return 'neutral', 0.0  # Return neutral if not trained yet

        try:
            text_vec = self.vectorizer.transform([text])
            prediction = self.logistic_model.predict(text_vec)[0]
            probabilities = self.logistic_model.predict_proba(text_vec)[0]
            return prediction, max(probabilities)
        except Exception as e:
            logger.error(f"Logistic prediction error: {e}")
            return 'neutral', 0.0

    def predict_textblob(self, text: str) -> Tuple[str, float]:
        """Fallback: TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1: return 'positive', abs(polarity)
            elif polarity < -0.1: return 'negative', abs(polarity)
            else: return 'neutral', abs(polarity)
        except: return 'neutral', 0.0

    def predict_huggingface(self, text: str) -> Tuple[str, float]:
        """Predict using HuggingFace (The Teacher)"""
        if not self.hf_model: return self.predict_textblob(text)
        try:
            # Truncate to 512 tokens to prevent crashes
            result = self.hf_model(text[:512])[0]
            label = result['label'].upper()
            if 'LABEL_1' in label or 'POSITIVE' in label: return 'positive', result['score']
            elif 'LABEL_0' in label or 'NEGATIVE' in label: return 'negative', result['score']
            return 'neutral', result['score']
        except: return self.predict_textblob(text)

    def analyze_article(self, article: Dict) -> Dict:
        """Main analysis pipeline"""
        content = article.get('processed_content', article.get('content', ''))
        if not content: return {'sentiment_label': 'neutral', 'sentiment_score': 0.0}
        
        # Use HF (Teacher) to get the label
        label, score = self.predict_huggingface(content)
        
        return {
            'sentiment_label': label,
            'sentiment_score': float(score),
            'model_used': 'huggingface'
        }