# src/database.py

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from typing import List, Dict, Optional
import logging
from config.config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class NewsDatabase:
    """Manages MongoDB operations for news articles"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DATABASE_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            
            # Create indexes for better query performance
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {Config.DATABASE_NAME}")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes"""
        try:
            # Unique index on URL to prevent duplicates
            self.collection.create_index([('url', ASCENDING)], unique=True)
            
            # Indexes for common queries
            self.collection.create_index([('source', ASCENDING)])
            self.collection.create_index([('scraped_at', DESCENDING)])
            self.collection.create_index([('sentiment_score', DESCENDING)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def insert_article(self, article: Dict) -> bool:
        """Insert a single article into the database"""
        try:
            self.collection.insert_one(article)
            logger.debug(f"Inserted article: {article.get('title', 'Unknown')}")
            return True
            
        except DuplicateKeyError:
            logger.debug(f"Article already exists: {article.get('url')}")
            return False
            
        except Exception as e:
            logger.error(f"Error inserting article: {e}")
            return False
    
    def insert_articles_bulk(self, articles: List[Dict]) -> int:
        """Insert multiple articles into the database"""
        inserted_count = 0
        
        for article in articles:
            if self.insert_article(article):
                inserted_count += 1
        
        logger.info(f"Inserted {inserted_count} new articles out of {len(articles)}")
        return inserted_count
    
    def update_article_sentiment(self, url: str, sentiment_data: Dict) -> bool:
        """Update sentiment analysis results for an article"""
        try:
            result = self.collection.update_one(
                {'url': url},
                {'$set': sentiment_data}
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating sentiment for {url}: {e}")
            return False
    
    def get_articles(self, filter_query: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
        """Retrieve articles from the database"""
        try:
            if filter_query is None:
                filter_query = {}
            
            articles = list(self.collection.find(filter_query).limit(limit))
            logger.info(f"Retrieved {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            return []
    
    def get_articles_by_source(self, source: str) -> List[Dict]:
        """Get articles from a specific source"""
        return self.get_articles({'source': source})
    
    def get_articles_without_sentiment(self) -> List[Dict]:
        """Get articles that haven't been analyzed for sentiment"""
        return self.get_articles({'sentiment_label': {'$exists': False}})
    
    def get_sentiment_statistics(self) -> Dict:
        """Get aggregate statistics on article sentiments"""
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$sentiment_label',
                        'count': {'$sum': 1},
                        'avg_score': {'$avg': '$sentiment_score'}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            stats = {
                'total_articles': self.collection.count_documents({}),
                'by_sentiment': results
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def get_source_statistics(self) -> Dict:
        """Get statistics grouped by news source"""
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$source',
                        'count': {'$sum': 1},
                        'avg_sentiment': {'$avg': '$sentiment_score'}
                    }
                },
                {'$sort': {'count': -1}}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            return {'by_source': results}
            
        except Exception as e:
            logger.error(f"Error calculating source statistics: {e}")
            return {}
    
    def delete_all_articles(self) -> int:
        """Delete all articles from the database"""
        try:
            result = self.collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} articles")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting articles: {e}")
            return 0
    
    def close(self):
        """Close the database connection"""
        self.client.close()
        logger.info("Database connection closed")