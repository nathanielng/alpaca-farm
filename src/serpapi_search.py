#!/usr/bin/env python

"""
SerpAPI Google Local Search client for finding local businesses.

A Python client library for interacting with the SerpAPI Google Local API. Provides tools for:
- General web search (https://serpapi.com/search-api)
- Local business searching
- Getting business details with ratings, reviews, and addresses
- Location-based searches
- Pagination through results

Requires authentication via SerpAPI key. Sign up at https://serpapi.com/
"""

import boto3
import json
import logging
import os
import requests
import sys

from botocore.config import Config
from datetime import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from typing import Optional, Dict, Any, List


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
# Validate API key on module load
if not SERPAPI_API_KEY:
    raise ValueError(
        "SERPAPI_API_KEY environment variable not set. "
        "Please set it in your .env file or provide it as a parameter."
    )


@tool
def search_google(query: str, api_key: Optional[str] = SERPAPI_API_KEY) -> Dict[str, Any]:
    """
    Search Google using SerpAPI and return the search results.

    Args:
        query: Search query (e.g., "Coffee", "Pizza near me", "Gas station")

    Returns:
        Dictionary containing search results with:
            - organic_results: List of organic search results
            - pagination: Pagination details

    Raises:
        requests.exceptions.RequestException: If the API request fails
        ValueError: If API key is missing or invalid

    Example:
        >>> results = search_google("Coffee shops near me")
        >>> for result in results.get('organic_results', []):
        ...     print(f"{result['title']}: {result['link']}")

    Reference:
        https://serpapi.com/search-api

    Note:
        - Requires SerpAPI key (paid service)
        - Returns up to 100 results per page
        - Use 'next' in pagination to get the next page of results
    """

    params = {
        "q": query,
        "engine": "google",
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results["organic_results"]
        return organic_results
    except Exception as e:
        logger.error(f"Error fetching search results: {e}")
        raise


@tool
def search_google_flights(departure_id: str, arrival_id: str, outbound_date: str, return_date: str, currency: str = "USD", api_key: Optional[str] = SERPAPI_API_KEY):
    """
    Search for flights using Google Flights via SerpAPI.

    Args:
        departure_id (str): Airport code for departure city (e.g. 'PEK')
        arrival_id (str): Airport code for arrival city (e.g. 'AUS') 
        outbound_date (str): Departure date in yyyy-mm-dd format
        return_date (str): Return date in yyyy-mm-dd format
        currency (str, optional): Currency code. Defaults to 'USD'
        hl (str, optional): Language code. Defaults to 'en'
        api_key (str, optional): SerpAPI key. If not provided, uses SERPAPI_API_KEY env var

    Returns:
        dict: Flight search results containing available flights, prices, airlines etc.

    Raises:
        Exception: If API request fails

    Reference:
        https://serpapi.com/google-flights-api
    """

    params = {
        "engine": "google_flights",
        "departure_id": departure_id,    # e.g. PEK
        "arrival_id": arrival_id,        # e.g. AUS
        "outbound_date": outbound_date,  # yyyy-mm-dd
        "return_date": return_date,      # yyyy-mm-dd
        "currency": currency,
        "hl": "en",
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    except Exception as e:
        logger.error(f"Error fetching search results: {e}")
        raise


@tool
def search_google_hotels(query: str, check_in_date: str, check_out_date: str, adults: int = 2, currency: str = "USD", api_key: Optional[str] = SERPAPI_API_KEY):
    """
    Search for hotels using Google Hotels via SerpAPI.

    Args:
        query (str): Hotel search query (e.g. 'Bali Resorts', 'Hotels in Paris')
        check_in_date (str): Check-in date in yyyy-mm-dd format
        check_out_date (str): Check-out date in yyyy-mm-dd format
        adults (int, optional): Number of adults. Defaults to 2
        currency (str, optional): Currency code. Defaults to 'USD'
        api_key (str, optional): SerpAPI key. If not provided, uses SERPAPI_API_KEY env var

    Returns:
        dict: Hotel search results containing available properties, prices, amenities etc.

    Raises:
        Exception: If API request fails

    Reference:
        https://serpapi.com/google-hotels-api
    """

    params = {
        "engine": "google_hotels",
        "q": query,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": str(adults),
        "currency": currency,
        "gl": "us",
        "hl": "en",
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    except Exception as e:
        logger.error(f"Error fetching search results: {e}")
        raise


@tool
def search_local_businesses(
    query: str,
    location: Optional[str] = None,
    gl: Optional[str] = None,
    hl: str = "en",
    start: int = 0,
    api_key: Optional[str] = SERPAPI_API_KEY
) -> Dict[str, Any]:
    """
    Search for local businesses using Google Local via SerpAPI.

    Args:
        query: Search query (e.g., "Coffee", "Pizza near me", "Gas station")
        location: Geographic location for search (e.g., "New York, NY", "Singapore")
        gl: Country code (e.g., "us", "sg", "uk")
        hl: Language code (default: "en")
        start: Pagination offset (default: 0). It skips the given number of results. It's used for pagination. On desktop, parameter only accepts multiples of 20 (e.g. 20 for the second page results, 40 for the third page results, etc.). On mobile, parameter only accepts multiples of 10 (e.g. 10 for the second page results, 20 for the third page results, etc.).
        api_key: Optional SerpAPI key. If not provided, will use SERPAPI_API_KEY env var

    Returns:
        Dictionary containing search results with:
            - local_results: List of local business results
            - ads_results: Sponsored listings
            - discover_more_places: Related search suggestions
            - pagination: Pagination details

    Raises:
        requests.exceptions.RequestException: If the API request fails
        ValueError: If API key is missing or invalid

    Example:
        >>> results = search_local_businesses("Coffee shops", location="Singapore")
        >>> for business in results.get('local_results', []):
        ...     print(f"{business['title']}: {business.get('rating', 'N/A')} stars")

    Reference:
        https://serpapi.com/google-local-api

    Note:
        - Requires SerpAPI key (paid service)
        - Returns up to 20 results per page
        - Use 'start' parameter for pagination
    """

    # Build query parameters
    params = {
        "engine": "google_local",
        "q": query,
        "location": location,
        "hl": hl,
        "start": start,
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        local_results = results["local_results"]
        logger.info(f"Successfully searched for '{query}' in {location or 'default location'}")
        return local_results
    except Exception as e:
        logger.error(f"Error fetching data from SerpAPI: {e}")
        raise


@tool
def search_google_maps(query: str, latitude: float, longitude: float, zoom: int = 14, api_key: Optional[str] = SERPAPI_API_KEY):
    """
    Search Google Maps using SerpAPI.

    Args:
        query (str): Search query (e.g. "Coffee", "Restaurants", "Hotels")
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate 
        api_key (str, optional): SerpAPI key. If not provided, uses SERPAPI_API_KEY env var

    Returns:
        dict: Google Maps search results containing places, businesses etc.

    Raises:
        Exception: If API request fails

    Reference:
        https://serpapi.com/google-maps-api
    """

    # ll parameter defines the GPS coordinates of the location where you want the search to originate from. Its value must match the following format:
    #   @ + latitude + , + longitude + , + zoom/map_height
    # This will form a string that looks like this:
    # e.g. @40.7455096,-74.0083012,14z or @43.8521864,11.2168835,10410m.
    # The zoom attribute ranges from 3z, map completely zoomed out - to 21z, map completely zoomed in. Alternatively, you can specify map_height in meters (e.g., 10410m).

    if zoom < 3 or zoom > 21:
        raise ValueError("Zoom level must be between 3 and 21")

    params = {
        "engine": "google_maps",
        "q": query,
        "ll": f"@{latitude},{longitude},{zoom}z",
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    except Exception as e:
        logger.error(f"Error fetching search results: {e}")
        raise


# AWS Bedrock configuration
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'us.amazon.nova-lite-v1:0')

# Create AWS session
session = boto3.Session()

# Create Bedrock model
model = BedrockModel(
    model_id=BEDROCK_MODEL_ID,
    max_tokens=2048,
    boto_client_config=Config(
        read_timeout=120,
        connect_timeout=120,
        retries=dict(max_attempts=3, mode="adaptive"),
    ),
    boto_session=session
)

SERPAPI_SYSTEM_PROMPT = """You are a helpful local business search assistant powered by Google Local via SerpAPI.

You can help users with:
1. Finding local businesses by type or name
2. Getting detailed information about businesses including ratings, reviews, and addresses
3. Searching for businesses near specific locations or coordinates
4. Finding business hours, contact information, and services offered
5. Searching for flights between airports
6. Finding and comparing hotel options
7. Exploring places on Google Maps

When users ask for information:
- Use search_local_businesses to find businesses by query and location
- Use search_google_flights to search for available flights
- Use search_google_hotels to find accommodations
- Use search_google_maps to explore specific locations
- Use search_google for general web searches

Always provide clear, helpful responses with relevant details from the search results.
Include business names, ratings, addresses, and contact information when available.
For flights, include prices, airlines, and flight times.
For hotels, include rates, amenities, and location details.
If multiple results are found, present them in a clear, organized format.
"""

serpapi_agent = Agent(
    model=model,
    system_prompt=SERPAPI_SYSTEM_PROMPT,
    tools=[
        search_google,
        search_google_flights,
        search_google_hotels,
        search_local_businesses,
        search_google_maps
    ]
)

EXAMPLE_PROMPTS = [
    "Find coffee shops in Singapore",
    "Search for flights from PEK to AUS next month",
    "Find hotels in Bali for 2 adults",
    "What are the top rated restaurants in Tokyo?", 
    "Search for pharmacies near latitude 40.7128 and longitude -74.0060",
    "Find shopping malls in Dubai with high ratings",
]


def run_interactive_agent():
    """
    Run the SerpAPI agent in interactive mode, taking user input from stdin.
    """
    print("=" * 70)
    print("SerpAPI Local Business Search Assistant")
    print("=" * 70)
    print("\nWelcome! I can help you with:")
    print("  • Find local businesses by name or type") 
    print("  • Get business ratings, reviews, and contact information")
    print("  • Search for businesses in specific locations")
    print("  • Find businesses near specific coordinates")
    print("  • Search for flights between airports")
    print("  • Find and compare hotel options")
    print("  • Explore places on Google Maps")
    print("\nType 'exit' or 'quit' to end the session.")
    print("=" * 70)
    print("\nExample queries:")
    for i, prompt in enumerate(EXAMPLE_PROMPTS, 1):
        print(f"  {i}. {prompt}")
    print("=" * 70)
    print()

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit commands 
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using SerpAPI Assistant. Goodbye!")
                break

            # Skip empty input
            if not user_input:
                continue

            # Send message to agent
            print("\nAssistant: ", end="", flush=True)

            # Add timeout handling
            try:
                response = serpapi_agent(user_input, timeout=60)
            except TimeoutError:
                print("\nRequest timed out. Please try again.")
                continue

            # Print the response
            print(response.content)

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            print(f"\nError: {e}")
            print("Please try again or type 'exit' to quit.")


def get_stdin():
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
    if sys.stdin.isatty():
        return ''  # Interactive terminal - no piped input
    else:
        return sys.stdin.read().strip()  # Input is being piped or redirected


def main(user_input):
    response = serpapi_agent(user_input)
    print(response)


if __name__ == "__main__":
    if not SERPAPI_API_KEY:
        print("Error: SERPAPI_API_KEY environment variable is required.")
        print("Please set it in your .env file or export it:")
        print("  export SERPAPI_API_KEY='your_api_key_here'")
        print("\nSign up for an API key at: https://serpapi.com/")
        sys.exit(1)

    user_input = get_stdin()
    if user_input:
        main(user_input)
    else:
        run_interactive_agent()
