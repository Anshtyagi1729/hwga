# src/scraper.py

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import logging
from typing import List, Dict
from config.config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class NewsScraper:
    """Web scraper for extracting news articles from various sources"""
    
    def __init__(self):
        self.headers = {'User-Agent': Config.USER_AGENT}
        self.session = requests.Session()
        
    def scrape_bbc(self, max_articles: int = Config.MAX_ARTICLES_PER_SOURCE) -> List[Dict]:
        """Scrape articles from BBC News"""
        articles = []
        try:
            logger.info("Scraping BBC News...")
            response = self.session.get(
                Config.NEWS_SOURCES['bbc'], 
                headers=self.headers, 
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # IMPORTANT: Using html.parser NOT lxml!
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links using multiple selectors
            article_links = []
            selectors = [
                'a[class*="promo"]',
                'h2 a',
                'h3 a',
                'article a'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                article_links.extend(links)
                if len(article_links) >= max_articles:
                    break
            
            # Remove duplicates
            seen_urls = set()
            unique_links = []
            for link in article_links:
                url = link.get('href', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_links.append(link)
            
            article_links = unique_links[:max_articles]
            logger.info(f"Found {len(article_links)} potential article links")
            
            for link in article_links:
                try:
                    url = link.get('href')
                    if not url:
                        continue
                        
                    # Make URL absolute
                    if url.startswith('/'):
                        url = 'https://www.bbc.com' + url
                    elif not url.startswith('http'):
                        continue
                    
                    # Skip non-article URLs
                    if any(skip in url for skip in ['#', 'javascript:', 'mailto:']):
                        continue
                    
                    title = link.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    
                    logger.info(f"Scraping: {title[:50]}...")
                    article_data = self._extract_article_content(url, 'bbc')
                    
                    if article_data and article_data.get('content'):
                        article_data.update({
                            'title': title,
                            'url': url,
                            'source': 'BBC',
                            'scraped_at': datetime.now()
                        })
                        articles.append(article_data)
                        logger.info(f"✅ Scraped: {title[:50]}...")
                        time.sleep(2)  # Be respectful
                        
                except Exception as e:
                    logger.warning(f"Error scraping BBC article: {e}")
                    continue
                        
        except Exception as e:
            logger.error(f"Error scraping BBC News: {e}")
            
        logger.info(f"Successfully scraped {len(articles)} articles from BBC")
        return articles
    
    def scrape_reuters(self, max_articles: int = Config.MAX_ARTICLES_PER_SOURCE) -> List[Dict]:
        """Scrape articles from Reuters (may be blocked)"""
        articles = []
        try:
            logger.info("Scraping Reuters...")
            response = self.session.get(
                Config.NEWS_SOURCES['reuters'], 
                headers=self.headers, 
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # IMPORTANT: Using html.parser NOT lxml!
            soup = BeautifulSoup(response.content, 'html.parser')
            article_containers = soup.find_all('article', limit=max_articles * 2)
            
            logger.info(f"Found {len(article_containers)} potential articles")
            
            for container in article_containers:
                if len(articles) >= max_articles:
                    break
                        
                try:
                    link = container.find('a')
                    if not link:
                        continue
                        
                    url = link.get('href')
                    if not url:
                        continue
                    
                    # Make URL absolute
                    if url.startswith('/'):
                        url = 'https.www.reuters.com' + url
                    elif not url.startswith('http'):
                        continue
                    
                    title = link.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    
                    logger.info(f"Scraping: {title[:50]}...")
                    article_data = self._extract_article_content(url, 'reuters')
                    
                    if article_data and article_data.get('content'):
                        article_data.update({
                            'title': title,
                            'url': url,
                            'source': 'Reuters',
                            'scraped_at': datetime.now()
                        })
                        articles.append(article_data)
                        logger.info(f"✅ Scraped: {title[:50]}...")
                        time.sleep(2)
                        
                except Exception as e:
                    logger.warning(f"Error scraping Reuters article: {e}")
                    continue
                        
        except Exception as e:
            logger.error(f"Error scraping Reuters: {e}")
            
        logger.info(f"Successfully scraped {len(articles)} articles from Reuters")
        return articles

    def scrape_custom_source(self, start_url: str, max_articles: int = 5) -> List[Dict]:
        """
        Generic scraper that attempts to find and scrape articles from any given URL.
        Uses heuristics to identify article links and content.
        """
        articles = []
        # Extract domain (e.g., 'cnn.com') from https://cnn.com/world
        try:
            domain = start_url.split('/')[2] 
        except IndexError:
            domain = "custom_source"
        
        try:
            logger.info(f"Custom scraping started for: {start_url}")
            response = self.session.get(start_url, headers=self.headers, timeout=Config.REQUEST_TIMEOUT)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 1. Find potential article links
            # We look for links that are longer than 25 chars (usually articles) 
            # and belong to the same domain.
            found_links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                
                # Fix relative URLs
                if href.startswith('/'):
                    href = f"{start_url.rstrip('/')}{href}"
                
                # Filter: Must be http/https, contain domain (basic check), and be long enough
                if href.startswith('http') and domain in href and len(href) > 25:
                    found_links.add(href)
                    
            # Limit the number of links to scrape
            target_links = list(found_links)[:max_articles]
            logger.info(f"Found {len(target_links)} potential articles on {domain}")
            
            # 2. Visit each link and extract content
            for link in target_links:
                try:
                    logger.info(f"Scraping generic: {link}")
                    art_response = self.session.get(link, headers=self.headers, timeout=10)
                    art_soup = BeautifulSoup(art_response.content, 'html.parser')
                    
                    # Generic Title Extraction (h1 usually)
                    title_tag = art_soup.find('h1')
                    title = title_tag.get_text(strip=True) if title_tag else art_soup.title.text
                    
                    # Generic Content Extraction (All paragraphs)
                    paragraphs = art_soup.find_all('p')
                    # Filter out short "menu" paragraphs
                    content = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
                    
                    if len(content) > 200: # Only save if we found substantial text
                        articles.append({
                            'title': title,
                            'url': link,
                            'content': content,
                            'source': domain, # Use domain as source name
                            'scraped_at': datetime.now()
                        })
                        time.sleep(1) # Be polite
                        
                except Exception as e:
                    logger.warning(f"Failed to parse generic article {link}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error connecting to custom source {start_url}: {e}")
            
        return articles
    
    def _extract_article_content(self, url: str, source: str) -> Dict:
        """Extract article content from URL"""
        try:
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # IMPORTANT: Using html.parser NOT lxml!
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paragraphs
            paragraphs = []
            
            if source == 'bbc':
                selectors = ['article p', 'main p', 'div[data-component="text-block"] p', 'p']
                for selector in selectors:
                    paragraphs = soup.select(selector)
                    if len(paragraphs) > 3:
                        break
                        
            elif source == 'reuters':
                selectors = ['article p', 'main p', 'div[class*="article"] p', 'p']
                for selector in selectors:
                    paragraphs = soup.select(selector)
                    if len(paragraphs) > 3:
                        break
            else:
                paragraphs = soup.find_all('p')
            
            # Extract text
            content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # Filter out short content
            if len(content) < 100:
                logger.warning(f"Content too short ({len(content)} chars)")
                return None
            
            # Extract publish date
            pub_date = None
            time_tag = soup.find('time')
            if time_tag:
                pub_date = time_tag.get('datetime')
            
            return {
                'content': content,
                'published_date': pub_date,
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return None
    
    def scrape_all_sources(self) -> List[Dict]:
        """Scrape articles from all configured sources"""
        all_articles = []
        
        # Try BBC
        try:
            bbc_articles = self.scrape_bbc()
            all_articles.extend(bbc_articles)
        except Exception as e:
            logger.error(f"Failed to scrape BBC: {e}")
        
        # Try Reuters (will likely fail with 401)
        # --- MODIFIED: Commented out to prevent 401 error ---
        # try:
        #     reuters_articles = self.scrape_reuters()
        #     all_articles.extend(reuters_articles)
        # except Exception as e:
        #     logger.error(f"Failed to scrape Reuters: {e}")
        
        logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles