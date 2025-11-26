# src/preprocessor.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import logging
from typing import List, Dict, Set
from config.config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")


class TextPreprocessor:
    """Handles text cleaning and preprocessing for sentiment analysis"""
    
    def __init__(self):
        """Initialize preprocessor with required resources"""
        # Load standard English stopwords
        base_stopwords = set(stopwords.words('english'))
        
        # CRITICAL FIX: Remove negation words from the stopword list.
        # We want to KEEP these because "not good" is very different from "good".
        self.negations = {'no', 'not', 'nor', 'neither', 'never', "didn't", "isn't", 
                         "wasn't", "aren't", "don't", "doesn't", "wouldn't", "couldn't"}
        self.stop_words = base_stopwords - self.negations
        
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # 1. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Remove Email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # IMPROVEMENT: We DO NOT lowercase everything or remove punctuation here anymore.
        # Modern AI models (like HuggingFace) need casing and punctuation to detect 
        # intensity (e.g., "BAD!" vs "bad").
        
        return text
    
    def preprocess_for_model(self, text: str) -> str:
        """
        Special preprocessing just for the AI Model.
        For HuggingFace, we want raw-ish text but without garbage.
        """
        return self.clean_text(text)

    def preprocess_for_analysis(self, text: str) -> str:
        """
        Heavy preprocessing for Word Clouds or Statistics (Bag of Words).
        Here we remove stopwords and lemmatize.
        """
        # Lowercase for statistics
        text = text.lower()
        
        # Remove special characters only for this specific statistical view
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        words = word_tokenize(text)
        
        # Remove stopwords (but keep negations!)
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Lemmatize (convert "running" -> "run")
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        
        return ' '.join(lemmatized_words)
    
    def get_basic_statistics(self, text: str) -> Dict:
        """Extract basic text statistics"""
        words = word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract most frequent keywords from text"""
        # Use the heavy preprocessing for keyword extraction
        processed_text = self.preprocess_for_analysis(text)
        words = word_tokenize(processed_text)
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def validate_article(self, article: Dict) -> bool:
        """Validate if article meets minimum requirements"""
        content = article.get('content', '')
        if not content:
            return False
        
        word_count = len(content.split())
        # Relaxed constraints
        if word_count < 20: return False # Too short
        
        return True
    
    def preprocess_articles(self, articles: List[Dict]) -> List[Dict]:
        """Preprocess a list of articles"""
        processed_articles = []
        
        for article in articles:
            if not self.validate_article(article):
                continue
            
            original_content = article.get('content', '')
            
            # Store two versions:
            # 1. processed_content: Cleaned but readable (for HF Model)
            # 2. clean_tokens: Heavily stripped (for Word Clouds/Stats)
            
            article['processed_content'] = self.preprocess_for_model(original_content)
            article['clean_tokens'] = self.preprocess_for_analysis(original_content)
            
            article['statistics'] = self.get_basic_statistics(original_content)
            article['keywords'] = self.extract_keywords(original_content)
            
            processed_articles.append(article)
        
        logger.info(f"Preprocessed {len(processed_articles)} articles")
        return processed_articles