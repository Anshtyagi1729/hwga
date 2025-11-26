# config/config.py

import os
import logging

class Config:
    """Global configuration settings for the News Sentiment Analysis System"""
    
    # ---------------------------------------------------------
    # DIRECTORY PATHS
    # ---------------------------------------------------------
    # Get the project root directory (moves up two levels from config.py)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Output directory for Visualizations (Must be inside 'static' for Flask to serve them)
    OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'images')
    
    # Directory for saving trained ML models
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # ---------------------------------------------------------
    # DATABASE SETTINGS
    # ---------------------------------------------------------
    # Connect to local MongoDB instance
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = 'news_sentiment_db'
    COLLECTION_NAME = 'articles'
    
    # ---------------------------------------------------------
    # SCRAPING SETTINGS
    # ---------------------------------------------------------
    NEWS_SOURCES = {
        'bbc': 'https://www.bbc.com/news',
        # Reuters often blocks scrapers, but we keep the URL here
        'reuters': 'https://www.reuters.com/world/'
    }
    
    # Headers to mimic a real browser request
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    REQUEST_TIMEOUT = 10
    MAX_ARTICLES_PER_SOURCE = 50
    
    # ---------------------------------------------------------
    # ANALYSIS SETTINGS
    # ---------------------------------------------------------
    MIN_ARTICLE_LENGTH = 50
    MAX_ARTICLE_LENGTH = 5000
    
    # HuggingFace Model for Sentiment Analysis
    HUGGINGFACE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # ---------------------------------------------------------
    # VISUALIZATION SETTINGS
    # ---------------------------------------------------------
    FIGURE_SIZE = (10, 6)
    DPI = 100
    
    # ---------------------------------------------------------
    # LOGGING SETTINGS
    # ---------------------------------------------------------
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        # Create image output directory
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        # Create models directory
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        
        # Create css directory (ensures structure exists for Flask)
        css_dir = os.path.join(cls.BASE_DIR, 'static', 'css')
        os.makedirs(css_dir, exist_ok=True)