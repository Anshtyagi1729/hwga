

import logging
import argparse
import pandas as pd
from config.config import Config
from src.scraper import NewsScraper
from src.database import NewsDatabase  # Changed from DatabaseManager
from src.preprocessor import TextPreprocessor
from src.sentiment import SentimentAnalyzer
from src.visualizer import SentimentVisualizer

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler('news_sentiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def scrape_news(db_manager: NewsDatabase, max_articles: int = 50):
    """Scrape news articles and store in database"""
    print("\n" + "=" * 80)
    print("STEP 1: SCRAPING NEWS ARTICLES")
    print("=" * 80)
    
    scraper = NewsScraper()
    articles = scraper.scrape_all_sources()
    
    if articles:
        # --- THIS LINE HAS BEEN CORRECTED ---
        inserted = db_manager.insert_articles_bulk(articles)
        print(f"✅ Successfully inserted {inserted} new articles")
    else:
        print("⚠️  No articles scraped")
    
    return len(articles)


# --- MODIFIED: Added 'analyzer' as an argument ---
def preprocess_and_analyze(db_manager: NewsDatabase, analyzer: SentimentAnalyzer):
    """Preprocess articles and perform sentiment analysis"""
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING AND SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Get articles from database
    # --- THIS LINE HAS BEEN CORRECTED ---
    articles = db_manager.get_articles(limit=0) # Set limit=0 to get all articles
    
    if not articles:
        print("⚠️  No articles found in database")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    # --- MODIFIED: Removed this line, we now pass the analyzer in ---
    # analyzer = SentimentAnalyzer() 
    
    # Train model and analyze
    metrics = analyzer.train_logistic_regression(df)
    
    # Update database with sentiment results
    print("\n📝 Updating database with sentiment results...")
    for idx, row in df.iterrows():
        sentiment_data = {
            'sentiment_label': int(row['sentiment_label']),
            'cleaned_title': row.get('cleaned_title', ''),
            'cleaned_content': row.get('cleaned_content', ''),
            'combined_text': row.get('combined_text', '')
        }
        # --- THIS LINE HAS BEEN CORRECTED ---
        # The database function updates using 'url', not '_id'
        db_manager.update_article_sentiment(row['url'], sentiment_data)
    
    print("✅ Database updated with sentiment analysis")
    
    return df, metrics


def visualize_results(db_manager: NewsDatabase, df, metrics):
    """Create visualizations from analyzed articles"""
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    visualizer = SentimentVisualizer()
    visualizer.generate_all_visualizations(df, metrics)


# --- THIS FUNCTION HAS BEEN CORRECTED ---
def display_statistics(db_manager: NewsDatabase):
    """Display database statistics"""
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    
    # Sentiment statistics
    sentiment_stats = db_manager.get_sentiment_statistics()
    print(f"\nTotal Articles: {sentiment_stats.get('total_articles', 0)}")
    
    if sentiment_stats.get('by_sentiment'):
        print("\nSentiment Breakdown:")
        for item in sentiment_stats['by_sentiment']:
            label = item.get('_id', 'Unknown')
            count = item.get('count', 0)
            # --- THIS LINE IS FIXED ---
            # Use 'or 0.0' to default to 0.0 if avg_score is None
            avg_score = item.get('avg_score') or 0.0 
            print(f"  {label}: {count} articles (avg score: {avg_score:.4f})")
    
    # Source statistics
    source_stats = db_manager.get_source_statistics()
    if source_stats.get('by_source'):
        print("\nArticles by Source:")
        for item in source_stats['by_source']:
            source = item.get('_id', 'Unknown')
            count = item.get('count', 0)
            # --- THIS LINE IS FIXED ---
            # Use 'or 0.0' to default to 0.0 if avg_sent is None
            avg_sent = item.get('avg_sentiment') or 0.0
            print(f"  {source}: {count} articles (avg sentiment: {avg_sent:.4f})")


# --- MODIFIED: Added 'analyzer' as an argument ---
def test_prediction(db_manager: NewsDatabase, analyzer: SentimentAnalyzer):
    """Test prediction on a sample article"""
    print("\n" + "=" * 80)
    print("TESTING PREDICTIONS")
    print("=" * 80)
    
    # Get a sample article
    # --- THIS LINE HAS BEEN CORRECTED ---
    articles = db_manager.get_articles(limit=1)
    
    if not articles:
        print("⚠️  No articles available for testing")
        return
    
    sample = articles[0]
    text = sample.get('combined_text', sample.get('content', ''))
    
    if not text:
        print("⚠️  No text content available")
        return
    
    # --- MODIFIED: Removed this line, we now pass the analyzer in ---
    # analyzer = SentimentAnalyzer()
    
    # Get predictions from both models
    print(f"\n📰 Sample Article Title: {sample.get('title', 'N/A')[:100]}...")
    print(f"📝 Text Preview: {text[:200]}...\n")
    
    results = analyzer.analyze_article(text)
    
    # Display Logistic Regression results
    lr_result = results['logistic_regression']
    print("🎯 Logistic Regression:")
    print(f"   Sentiment: {lr_result['sentiment']}")
    print(f"   Confidence: {lr_result['confidence']:.2%}")
    print(f"   Probabilities: Neg={lr_result['probabilities']['negative']:.2%}, "
          f"Pos={lr_result['probabilities']['positive']:.2%}")
    
    # Display Hugging Face results
    hf_result = results['hugging_face']
    if 'error' not in hf_result:
        print("\n🤗 Hugging Face (DistilBERT):")
        print(f"   Sentiment: {hf_result['sentiment']}")
        print(f"   Confidence: {hf_result['confidence']:.2%}")
    else:
        print(f"\n⚠️  Hugging Face: {hf_result['error']}")


def run_full_pipeline(args):
    """Run the complete sentiment analysis pipeline"""
    print("\n" + "=" * 80)
    print("NEWS SENTIMENT ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Setup directories
    Config.setup_directories()
    
    # Initialize database
    db_manager = NewsDatabase()
    
    # --- ADDED: Create ONE analyzer for the whole pipeline ---
    analyzer = SentimentAnalyzer()
    
    try:
        df = None
        metrics = None
        
        # Step 1: Scrape (if requested)
        if args.scrape:
            scrape_news(db_manager, max_articles=args.max_articles)
        
        # Step 2: Preprocess and Analyze (if requested)
        if args.analyze:
            # --- MODIFIED: Pass the analyzer instance ---
            df, metrics = preprocess_and_analyze(db_manager, analyzer)
        
        # Step 3: Visualize (if requested and we have data)
        if args.visualize:
            if df is None or metrics is None:
                # Need to get data from database
                # --- THIS LINE HAS BEEN CORRECTED ---
                articles = db_manager.get_articles(limit=0)
                if articles:
                    df = pd.DataFrame(articles)
                    if 'sentiment_label' in df.columns:
                        print("📊 Using existing sentiment data for visualization")
                        # Create dummy metrics for visualization
                        metrics = {'confusion_matrix': None}
                        visualizer = SentimentVisualizer()
                        # --- FIX: Pass df to visualization functions ---
                        visualizer.plot_sentiment_distribution(df)
                        visualizer.generate_wordcloud(df, sentiment_label=1)
                        visualizer.generate_wordcloud(df, sentiment_label=0)
                    else:
                        print("⚠️  No sentiment data available. Run with --analyze first.")
                else:
                    print("⚠️  No articles in database for visualization")
            else:
                visualize_results(db_manager, df, metrics)
        
        # Step 4: Test predictions
        if args.test:
            # --- MODIFIED: Pass the analyzer instance ---
            test_prediction(db_manager, analyzer)
        
        # Display statistics
        display_statistics(db_manager)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_manager.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='News Sentiment Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all           # Run complete pipeline
  python main.py --scrape         # Only scrape articles
  python main.py --analyze         # Only analyze sentiment
  python main.py --visualize       # Only create visualizations
  python main.py --scrape --analyze # Scrape and analyze
  python main.py --test           # Test predictions on sample
        """
    )
    
    # Pipeline steps
    parser.add_argument('--scrape', action='store_true', 
                        help='Scrape news articles from sources')
    parser.add_argument('--analyze', action='store_true',
                        help='Preprocess and perform sentiment analysis')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--test', action='store_true',
                        help='Test predictions on sample articles')
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (scrape, analyze, visualize)')
    
    # Options
    parser.add_argument('--max-articles', type=int, default=50,
                        help='Maximum articles to scrape per source (default: 50)')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all steps
    if args.all:
        args.scrape = True
        args.analyze = True
        args.visualize = True
        args.test = True
    
    # If no arguments specified, show help
    if not (args.scrape or args.analyze or args.visualize or args.test):
        parser.print_help()
        print("\n⚠️  No action specified. Use --all to run complete pipeline or specify individual steps.")
        return
    
    # Run pipeline
    run_full_pipeline(args)


if __name__ == '__main__':
    main()