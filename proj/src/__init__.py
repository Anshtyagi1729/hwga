# src/__init__.py

"""
News Sentiment Analysis System
A comprehensive system for scraping, analyzing, and visualizing news article sentiments
"""

from .scraper import NewsScraper
# FIXED: Changed 'DatabaseManager' to 'NewsDatabase' to match your database.py file
from .database import NewsDatabase  
from .preprocessor import TextPreprocessor
from .sentiment import SentimentAnalyzer
from .visualizer import SentimentVisualizer

__version__ = '1.0.0'
__author__ = 'rohank'

__all__ = [
    'NewsScraper',
    'NewsDatabase',  # FIXED: Updated this list as well
    'TextPreprocessor',
    'SentimentAnalyzer',
    'SentimentVisualizer'
]