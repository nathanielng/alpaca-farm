#!/usr/bin/env python

"""
Usage:
  echo "https://aws.amazon.com/" | web2markdown.py
  echo "https://aws.amazon.com/" | web2markdown.py --engine firecrawl
  echo "https://aws.amazon.com/" | web2markdown.py --engine playwright
  echo "https://aws.amazon.com/" | web2markdown.py --engine textfromwebsite

Installation:
  uv pip install html2text
  playwright install
"""

import argparse
import logging
import os
import requests
import sys
import time

from playwright.sync_api import sync_playwright
from html2text import HTML2Text


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", '')


def html_to_text(url: str) -> str:
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        html = requests.get(url).text
        return h.handle(html)
    except ImportError:
        raise "Please install html2text using 'pip install html2text'"


def firecrawl_to_markdown(url, api_key = FIRECRAWL_API_KEY) -> str:
    """
    Use Firecrawl API for professional-grade scraping.
    Handles JS, anti-bot measures, and returns clean markdown.
    """
    if not api_key:
        raise ValueError("API key not provided and FIRECRAWL_API_KEY environment variable not set")
    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=api_key)
        formats = ['markdown']  # ['markdown', 'html']
        response = app.scrape(url, formats=formats)  # <class 'firecrawl.v2.types.Document'>
        return response.markdown
    except ImportError:
        raise "Please install firecrawl using 'pip install firecrawl'"


def playwright_to_markdown(url: str, wait_time: int = 2):
    """
    Use Playwright to render JavaScript and convert to markdown.
    Perfect for SPAs, React apps, and dynamic content.
    """
    with sync_playwright() as p:
        # Launch browser (use headless=False to see what's happening)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to the URL
        print(f"Loading {url}...")
        page.goto(url, wait_until="networkidle")
        
        # Wait for dynamic content to load
        time.sleep(wait_time)
        
        # Optional: Scroll to load lazy-loaded content
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        
        # Get the fully rendered HTML
        html = page.content()
        
        browser.close()
        
        # Convert HTML to markdown
        h = HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0
        
        markdown = h.handle(html)
        
        return markdown


def playwright_with_interactions(url: str, selector: str = ".article-content", timeout: int = 5000):
    """
    Advanced example: Click buttons, fill forms, etc.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        page.goto(url)
        
        # Example: Click "Load More" button if it exists
        try:
            load_more = page.locator("button:has-text('Load More')")
            if load_more.is_visible():
                load_more.click()
                page.wait_for_load_state("networkidle")
        except:
            pass
        
        # Example: Wait for specific element
        try:
            page.wait_for_selector(selector, timeout=timeout)
        except:
            print("Timeout waiting for content")
        
        html = page.content()
        browser.close()
        
        # Convert to markdown
        h = HTML2Text()
        h.ignore_links = False
        markdown = h.handle(html)
        
        return markdown


def download_url_content(url: str) -> str:
    """
    Downloads content from a URL and returns it as a string
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def text_from_website(url: str) -> str:
    """
    Args:
        url: string beginning with http://
    Guide:
        https://textfrom.website/
    """
    try:
        return download_url_content(f"https://textfrom.website/{url}")
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        raise


def main(user_input, engine):
    if engine == 'html2text':
        print(html_to_text(user_input))
    elif engine == 'firecrawl':
        print(firecrawl_to_markdown(user_input))
    elif engine == 'playwright':
        print(playwright_to_markdown(user_input))
    elif engine == 'textfromwebsite':
        print(text_from_website(user_input))
    else:
        raise ValueError("Please specify an engine: html2text, firecrawl, or playwright")


def get_stdin():
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
    if sys.stdin.isatty():
        return ''  # Interactive terminal - no piped input
    else:
        return sys.stdin.read().strip()  # Input is being piped or redirected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a webpage to markdown.")
    parser.add_argument("--engine", choices=['html2text', 'firecrawl', 'playwright', 'textfromwebsite'],
                        default='html2text',
                        help="Specify the engine to use for conversion")
    args = parser.parse_args()

    user_input = get_stdin()
    if user_input:
        main(user_input, args.engine)
    else:
        print("No input provided")
