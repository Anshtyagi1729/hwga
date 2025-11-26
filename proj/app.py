from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import logging
from datetime import datetime

# Import your custom modules
from src.database import NewsDatabase
from src.scraper import NewsScraper
from src.sentiment import SentimentAnalyzer
from src.preprocessor import TextPreprocessor
from src.visualizer import SentimentVisualizer
from config.config import Config

# Configure Logging
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'  # Change this for production

# Initialize System Components
# We do this globally so connections stay open
try:
    Config.setup_directories()  # Ensure static/images exists
    db_manager = NewsDatabase()
    scraper = NewsScraper()
    analyzer = SentimentAnalyzer()
    
    # --- NEW: Train Logistic Regression on startup if data exists ---
    logger.info("Checking for existing data to train Logistic Regression model...")
    existing_data = db_manager.get_articles({'sentiment_label': {'$exists': True}}, limit=0)
    if existing_data:
        train_msg = analyzer.train_on_db(existing_data)
        logger.info(f"Startup Training: {train_msg}")
    else:
        logger.info("No labeled data found for startup training. Model will start empty.")
    # ----------------------------------------------------------------
    
    preprocessor = TextPreprocessor()
    visualizer = SentimentVisualizer()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")

# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------

@app.route('/')
def index():
    """Home page: Shows controls, stats, and ALL articles"""
    try:
        # Get statistics
        stats = db_manager.get_sentiment_statistics()
        
        # Get all articles to verify scraping
        recent_articles = db_manager.get_articles(limit=0)
        
        return render_template('index.html', 
                             stats=stats, 
                             articles=recent_articles)
    except Exception as e:
        flash(f"Error loading dashboard: {e}", "danger")
        return render_template('index.html', stats={}, articles=[])


@app.route('/scrape', methods=['POST'])
def scrape():
    """Trigger the news scraper (BBC/Reuters)"""
    try:
        logger.info("Manual scrape triggered from UI")
        articles = scraper.scrape_all_sources()
        
        if articles:
            count = db_manager.insert_articles_bulk(articles)
            flash(f"✅ Successfully scraped and saved {count} new articles!", "success")
        else:
            flash("⚠️ Scraper ran but found no new articles.", "warning")
            
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        flash(f"❌ Error during scraping: {str(e)}", "danger")
        
    return redirect(url_for('index'))


@app.route('/scrape_custom', methods=['POST'])
def scrape_custom():
    """Handle custom URL scraping"""
    url = request.form.get('custom_url')
    
    if not url:
        flash("⚠️ Please enter a valid URL", "warning")
        return redirect(url_for('index'))
    
    # Add http if missing
    if not url.startswith('http'):
        url = 'https://' + url
        
    try:
        flash(f"🕵️ Starting scan of {url}... This might take a moment.", "info")
        
        # Call the scraper
        articles = scraper.scrape_custom_source(url, max_articles=5)
        
        if articles:
            # Save to database
            count = db_manager.insert_articles_bulk(articles)
            flash(f"✅ Found and saved {count} articles from {url}!", "success")
            
            # --- THIS IS THE KEY CHANGE ---
            # We fetch the stats and main list again, BUT we also pass 'scraped_articles'
            stats = db_manager.get_sentiment_statistics()
            recent_articles = db_manager.get_articles(limit=0)
            
            return render_template('index.html', 
                                 stats=stats, 
                                 articles=recent_articles,
                                 scraped_articles=articles) # <--- Sending the new data to the UI
        else:
            flash(f"⚠️ Could not find readable articles on {url}.", "warning")
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Custom scrape error: {e}")
        flash(f"❌ Error scraping URL: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    """Trigger sentiment analysis on unanalyzed articles"""
    try:
        # 1. Get all articles (or you could filter for just unanalyzed ones)
        articles = db_manager.get_articles(limit=0)
        
        if not articles:
            flash("⚠️ No articles found in database to analyze.", "warning")
            return redirect(url_for('index'))

        # 2. Preprocess
        df = pd.DataFrame(articles)
        
        # Only process if we have content
        if 'content' not in df.columns:
            flash("⚠️ Articles found but they have no content to analyze.", "warning")
            return redirect(url_for('index'))

        logger.info(f"Analyzing {len(df)} articles...")
        
        # 3. Analyze and Update Database
        count = 0
        for _, row in df.iterrows():
            # Skip if already analyzed (optional optimization)
            # if 'sentiment_label' in row and pd.notna(row['sentiment_label']): continue

            # Text to analyze: use processed content if available, else raw
            text_to_analyze = row.get('content', '') # CHANGED to use raw content for better accuracy
            
            # Perform Analysis
            result = analyzer.analyze_article({'content': text_to_analyze})
            
            # Update Database
            update_data = {
                'sentiment_label': result['sentiment_label'],
                'sentiment_score': result['sentiment_score'],
                'model_used': result['model_used']
            }
            
            success = db_manager.update_article_sentiment(row['url'], update_data)
            if success:
                count += 1
        
        # --- NEW: Retrain Logistic Regression with the freshly analyzed data ---
        if count > 0:
            logger.info("Analysis complete. Retraining Logistic Regression model...")
            all_labeled_data = db_manager.get_articles({'sentiment_label': {'$exists': True}}, limit=0)
            train_msg = analyzer.train_on_db(all_labeled_data)
            flash(f"✅ Analysis complete! Updated {count} articles. ({train_msg})", "success")
        else:
            flash(f"✅ Analysis complete! Updated {count} articles.", "success")
        # -----------------------------------------------------------------------

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        flash(f"❌ Error during analysis: {str(e)}", "danger")
        
    return redirect(url_for('index'))


@app.route('/visualize')
def dashboard():
    """Generate plots and show the dashboard"""
    try:
        articles = db_manager.get_articles(limit=0)
        
        if not articles:
            flash("⚠️ No data available to visualize.", "warning")
            return redirect(url_for('index'))

        # Check if we have sentiment data
        if 'sentiment_label' not in articles[0]:
             flash("⚠️ Data exists but hasn't been analyzed yet. Run 'Analysis' first.", "warning")
             # We still render the page, but images might be empty/broken
        
        # Generate Visualizations (Saves them to static/images)
        # This ensures the dashboard always shows fresh data
        visualizer.generate_all_visualizations(articles)
        
        return render_template('dashboard.html')
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        flash(f"❌ Error generating visualizations: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/test_prediction', methods=['POST'])
def test_prediction():
    """Handle the single-text prediction form"""
    text = request.form.get('text_input', '').strip()
    
    if not text:
        flash("⚠️ Please enter some text to analyze.", "warning")
        return redirect(url_for('index'))
    
    try:
        # Get predictions from both models
        hf_label, hf_score = analyzer.predict_huggingface(text)
        lr_label, lr_score = analyzer.predict_logistic(text)
        
        result = {
            'text': text,
            'hf': (hf_label, hf_score),
            'lr': (lr_label, lr_score)
        }
        
        # Re-render index with the result
        stats = db_manager.get_sentiment_statistics()
        # MODIFIED: Ensure we get all articles for the list
        recent_articles = db_manager.get_articles(limit=0)
        
        return render_template('index.html', 
                             prediction_result=result,
                             stats=stats,
                             articles=recent_articles)
                             
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        flash(f"❌ Error processing text: {str(e)}", "danger")
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' makes it accessible on your local network
    app.run(debug=True, host='0.0.0.0', port=5000)