# src/visualizer.py

import matplotlib
matplotlib.use('Agg')  # Fixes the main thread error!
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
# from wordcloud import WordCloud  <-- Commented out
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from config.config import Config
import os

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SentimentVisualizer:
    """Create visualizations for sentiment analysis results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.output_dir = Config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_sentiment_distribution(self, articles: List[Dict], 
                                   save_path: str = None) -> None:
        """Plot distribution of sentiment labels"""
        sentiments = [article.get('sentiment_label', 'neutral') for article in articles]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)
        
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        bar_colors = [colors.get(label, '#95a5a6') for label in sentiment_counts.index]
        
        sentiment_counts.plot(kind='bar', ax=ax, color=bar_colors)
        ax.set_title('Sentiment Distribution Across Articles', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sentiment_distribution.png')
        
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Sentiment distribution plot saved to {save_path}")
        plt.close()
    
    def plot_sentiment_by_source(self, articles: List[Dict], 
                                save_path: str = None) -> None:
        """Plot sentiment distribution by news source"""
        df = pd.DataFrame(articles)
        
        if 'source' not in df.columns or 'sentiment_label' not in df.columns:
            logger.warning("Missing required columns for source plot")
            return
        
        # Create cross-tabulation
        ct = pd.crosstab(df['source'], df['sentiment_label'])
        
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)
        ct.plot(kind='bar', stacked=True, ax=ax, 
               color=['#2ecc71', '#e74c3c', '#95a5a6'])
        
        ax.set_title('Sentiment Distribution by News Source', fontsize=16, fontweight='bold')
        ax.set_xlabel('News Source', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sentiment_by_source.png')
        
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Sentiment by source plot saved to {save_path}")
        plt.close()
    
    def plot_sentiment_scores(self, articles: List[Dict], 
                             save_path: str = None) -> None:
        """Plot distribution of sentiment scores"""
        scores = [article.get('sentiment_score', 0.5) for article in articles]
        
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)
        
        ax.hist(scores, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_title('Distribution of Sentiment Confidence Scores', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.axvline(np.mean(scores), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sentiment_scores.png')
        
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Sentiment scores plot saved to {save_path}")
        plt.close()
    
    # REMOVED WORD CLOUD METHOD

    def plot_interactive_timeline(self, articles: List[Dict]) -> None:
        """Create interactive timeline of sentiment over time"""
        df = pd.DataFrame(articles)
        
        if 'scraped_at' not in df.columns:
            logger.warning("No timestamp data available for timeline")
            return
        
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df = df.sort_values('scraped_at')
        
        # Create interactive plot
        fig = px.scatter(df, x='scraped_at', y='sentiment_score',
                        color='sentiment_label',
                        hover_data=['title', 'source'],
                        title='Sentiment Timeline',
                        color_discrete_map={
                            'positive': '#2ecc71',
                            'negative': '#e74c3c',
                            'neutral': '#95a5a6'
                        })
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Sentiment Score',
            hovermode='closest'
        )
        
        save_path = os.path.join(self.output_dir, 'sentiment_timeline.html')
        fig.write_html(save_path)
        logger.info(f"Interactive timeline saved to {save_path}")
    
    def create_summary_report(self, articles: List[Dict]) -> None:
        """Create comprehensive summary report"""
        df = pd.DataFrame(articles)
        
        # Calculate statistics
        total_articles = len(articles)
        sentiment_counts = df['sentiment_label'].value_counts()
        avg_score = df['sentiment_score'].mean()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Sentiment Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sentiment_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%',
                            colors=['#e74c3c','#2ecc71', '#95a5a6'])
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        ax1.set_ylabel('')
        
        # 2. Sentiment by Source
        ax2 = fig.add_subplot(gs[0, 1])
        if 'source' in df.columns:
            ct = pd.crosstab(df['source'], df['sentiment_label'])
            ct.plot(kind='bar', ax=ax2, stacked=False)
            ax2.set_title('Sentiment by Source', fontweight='bold')
            ax2.set_xlabel('Source')
            ax2.set_ylabel('Count')
            ax2.legend(title='Sentiment')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Score Distribution
        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(df['sentiment_score'], bins=30, color='#3498db', 
                edgecolor='black', alpha=0.7)
        ax3.set_title('Sentiment Score Distribution', fontweight='bold')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.axvline(avg_score, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {avg_score:.3f}')
        ax3.legend()
        
        # 4. Summary Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        summary_text = f"""
        Summary Statistics
        {'=' * 50}
        Total Articles Analyzed: {total_articles}
        Average Sentiment Score: {avg_score:.4f}
        
        Sentiment Breakdown:
        {'-' * 50}
        """
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_articles) * 100
            summary_text += f"\n        {sentiment.capitalize()}: {count} ({percentage:.1f}%)"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('News Sentiment Analysis - Summary Report', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        save_path = os.path.join(self.output_dir, 'summary_report.png')
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Summary report saved to {save_path}")
        plt.close()
    
    def generate_all_visualizations(self, articles: List[Dict]) -> None:
        """Generate all available visualizations"""
        logger.info("Generating all visualizations...")
        
        self.plot_sentiment_distribution(articles)
        self.plot_sentiment_by_source(articles)
        self.plot_sentiment_scores(articles)
        # self.create_wordcloud(articles, sentiment='all')
        # self.create_wordcloud(articles, sentiment='positive')
        # self.create_wordcloud(articles, sentiment='negative')
        self.plot_interactive_timeline(articles)
        self.create_summary_report(articles)
        
        logger.info(f"All visualizations saved to {self.output_dir}")