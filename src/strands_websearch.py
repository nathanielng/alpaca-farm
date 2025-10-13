#!/usr/bin/env python

"""
Installaiton:
    uv pip install duckduckgo_search exa-py tavily-python
Usage:
    echo "Tell me about search engines" | strands_websearch.py --engine duckduckgo
"""

import argparse
import logging
import os
import sys

from botocore.config import Config
from strands import Agent, tool
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.models.bedrock import BedrockModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logging.getLogger("strands").setLevel(logging.INFO)

HOME = os.getenv('HOME')
BEDROCK_REGION = os.getenv("BEDROCK_REGION", 'us-west-2')
BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"

# ----- Tavily Search -----
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', None)
TAVILY_SYSTEM_PROMPT = """
You are a search assistant with access to the Tavily API.
You can:
1. Search the internet with a query

When displaying responses:
- Format data in a human-readable way
- Highlight important information
- Include source URLs and keep findings under 500 words
"""

# ----- Exa Search -----
EXA_API_KEY = os.getenv('EXA_API_KEY', None)
EXA_SYSTEM_PROMPT = """
You are a search assistant with access to the Exa API.
You can:
1. Search the internet with a query
2. Filter results by relevance and credibility 
3. Extract key information from multiple sources

When displaying responses:
- Format data in a clear, structured way
- Highlight important information with bullet points
- Include source URLs and publication dates
- Keep findings under 500 words
- Prioritize recent and authoritative sources
"""

# ----- Duck Duck Go Search -----
DUCKDUCKGO_SYSTEM_PROMPT = """
You are a search assistant with access to the Duck Duck Go API.
You can:
1. Search the internet with a query

When displaying responses:
- Format data in a human-readable way
- Highlight important information
- Include source URLs and keep findings under 500 words
"""

@tool
def tavily_search(query: str, max_results: int = 3):
    """
    Perform an internet search with the specified query using Tavily API
    
    Args:
        query: A question or search phrase to perform a search with
        max_results: Maximum number of search results to return (default: 3)
        
    Returns:
        Search results containing relevant web pages from Tavily
    """
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        return tavily_client.search(
            query,
            max_results = max_results
        )
    except ImportError:
        logger.error("Tavily is not installed. Please install it using 'pip install tavily-python'.")
        return None

@tool
def exa_search(query: str, max_results: int = 3):
    """
    Perform an internet search with the specified query using Exa API
    
    Args:
        query: A question or search phrase to perform a search with
        max_results: Maximum number of search results to return (default: 3)
        
    Returns:
        Search results containing relevant web pages from Exa
    """
    try:
        from exa_py import Exa
        exa = Exa(EXA_API_KEY)
        return exa.search_and_contents(
            query,
            type="auto",
            text=True,
            num_results = max_results
        )
    except ImportError:
        logger.error("Exa is not installed. Please install it using 'pip install exa-py'.")
        return None

@tool
def duckduckgo_search(query: str, max_results: int = 3):
    """
    Perform an internet search with the specified query using Duck Duck Go
    
    Args:
        query: A question or search phrase to perform a search with
        max_results: Maximum number of search results to return (default: 3)
        
    Returns:
        Search results containing relevant web pages from Duck Duck Go
    """

    try:
        from duckduckgo_search import DDGS
        response = DDGS().text(
            query,
            max_results = max_results
        )
        return response
    except ImportError:
        logger.error("Duck Duck Go is not installed. Please install it using 'pip install duckduckgo-search'.")
        return None

# Initialize Strands Agent
model = BedrockModel(
    model_id = BEDROCK_MODEL_ID,
    max_tokens = 2048,
    boto_client_config = Config(
        read_timeout = 120,
        connect_timeout = 120,
        retries = dict(max_attempts=3, mode="adaptive"),
    ),
    temperature = 0.1
)

def main(user_input, engine):

    if engine == 'exa':
        if EXA_API_KEY is None:
            logger.error("Please set the EXA_API_KEY environment variable.")
            raise
        SYSTEM_PROMPT = EXA_SYSTEM_PROMPT
        tools = [exa_search]
    elif engine == 'tavily':
        if TAVILY_API_KEY is None:
            logger.error("Please set the TAVILY_API_KEY environment variable.")
            raise
        SYSTEM_PROMPT = TAVILY_SYSTEM_PROMPT
        tools = [tavily_search]
    elif engine == 'duckduckgo':
        SYSTEM_PROMPT = DUCKDUCKGO_SYSTEM_PROMPT
        tools = [duckduckgo_search]
    else:
        logger.error(f"Unknown engine: {engine}")
        return

    agent = Agent(
        system_prompt = SYSTEM_PROMPT,
        model = model,
        tools = tools,
        callback_handler = PrintingCallbackHandler()
    )
    print(agent(user_input))


def get_stdin():
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
    if sys.stdin.isatty():
        return ''  # Interactive terminal - no piped input
    else:
        return sys.stdin.read().strip()  # Input is being piped or redirected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Strands")
    parser.add_argument('--engine', type=str, default='exa', choices=['exa', 'tavily', 'duckduckgo'], help="Search engine to use")
    args = parser.parse_args()

    user_input = get_stdin()
    if user_input:
        main(user_input, args.engine)
    else:
        print("No input provided")
