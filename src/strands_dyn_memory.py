#!/usr/bin/env python

"""
Memory Storage Agent - Stores information in DynamoDB as memory items

Usage:
  echo "Store this web page: https://example.com" | python memory_agent.py
  echo "Remember that Python 3.12 was released in October 2023" | python memory_agent.py

The agent will:
1. Detect if input contains a URL and fetch its content
2. Generate a TLDR and hashtags
3. Store in DynamoDB with timestamp and metadata
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from decimal import Decimal

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
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

# Configuration
HOME = os.getenv('HOME')
BEDROCK_REGION = os.getenv("BEDROCK_REGION", 'us-west-2')
BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"
DYNAMODB_REGION = os.getenv("DYNAMODB_REGION", BEDROCK_REGION)
MEMORY_TABLE_NAME = os.getenv("MEMORY_TABLE_NAME", "AIMemories")

# Initialize AWS clients
dynamodb_client = boto3.client('dynamodb', region_name=DYNAMODB_REGION)
dynamodb_resource = boto3.resource('dynamodb', region_name=DYNAMODB_REGION)

# Initialize Strands Model
model = BedrockModel(
    model_id=BEDROCK_MODEL_ID,
    max_tokens=2048,
    boto_client_config=Config(
        read_timeout=120,
        connect_timeout=120,
        retries=dict(max_attempts=3, mode="adaptive"),
    ),
    temperature=0.3
)

SYSTEM_PROMPT = """You are a memory storage assistant. Your job is to:
1. Process user input to extract key information
2. If a URL is mentioned, fetch and analyze the web content
3. Create concise summaries (TLDR) and relevant hashtags
4. Store the information in DynamoDB for later retrieval

Be concise, accurate, and organize information in a structured way."""


# ============================================================================
# Tools
# ============================================================================

@tool
def fetch_webpage_content(url: str) -> str:
    """
    Fetch webpage content and convert to markdown using web2markdown.py.
    
    Args:
        url: The URL to fetch
    
    Returns:
        Markdown content of the webpage or error message
    """
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web2markdown_path = os.path.join(script_dir, 'web2markdown.py')
        
        if not os.path.exists(web2markdown_path):
            return f"Error: web2markdown.py not found at {web2markdown_path}"
        
        # Use jina engine by default (no external dependencies needed)
        result = subprocess.run(
            ['python', web2markdown_path, '--engine', 'jina'],
            input=url,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            content = result.stdout.strip()
            # Limit content length to avoid token limits
            if len(content) > 10000:
                content = content[:10000] + "\n\n[Content truncated...]"
            return content
        else:
            return f"Error fetching webpage: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "Error: Timeout while fetching webpage"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def ensure_memory_table_exists() -> str:
    """
    Check if the memory table exists, create it if it doesn't.
    
    Returns:
        Status message
    """
    try:
        # Check if table exists
        dynamodb_client.describe_table(TableName=MEMORY_TABLE_NAME)
        return f"Memory table '{MEMORY_TABLE_NAME}' already exists"
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Table doesn't exist, create it
            try:
                dynamodb_client.create_table(
                    TableName=MEMORY_TABLE_NAME,
                    KeySchema=[
                        {'AttributeName': 'memory_id', 'KeyType': 'HASH'},
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'memory_id', 'AttributeType': 'S'},
                    ],
                    BillingMode='PAY_PER_REQUEST'  # On-demand pricing
                )
                
                # Wait for table to be created
                waiter = dynamodb_client.get_waiter('table_exists')
                waiter.wait(TableName=MEMORY_TABLE_NAME)
                
                logger.info(f"Created memory table: {MEMORY_TABLE_NAME}")
                return f"Successfully created memory table '{MEMORY_TABLE_NAME}'"
            
            except Exception as create_error:
                return f"Error creating table: {str(create_error)}"
        else:
            return f"Error checking table: {e.response['Error']['Message']}"


@tool
def store_memory_item(
    content: str,
    tldr: str,
    hashtags: list,
    source_url: str = None,
    metadata: dict = None
) -> str:
    """
    Store a memory item in DynamoDB.
    
    Args:
        content: The full content to store
        tldr: A concise summary (1-3 sentences)
        hashtags: List of relevant hashtags
        source_url: Optional URL if this memory came from a webpage
        metadata: Optional additional metadata
    
    Returns:
        Success message with memory_id
    """
    try:
        table = dynamodb_resource.Table(MEMORY_TABLE_NAME)
        
        # Generate unique memory ID
        timestamp = datetime.utcnow()
        memory_id = f"mem_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare item
        item = {
            'memory_id': memory_id,
            'content': content,
            'tldr': tldr,
            'hashtags': hashtags,
            'created_at': timestamp.isoformat(),
            'timestamp': int(timestamp.timestamp())
        }
        
        if source_url:
            item['source_url'] = source_url
        
        if metadata:
            item['metadata'] = _convert_floats_to_decimal(metadata)
        
        # Store in DynamoDB
        table.put_item(Item=item)
        
        logger.info(f"Stored memory: {memory_id}")
        return f"Successfully stored memory with ID: {memory_id}\nTLDR: {tldr}\nHashtags: {', '.join(hashtags)}"
    
    except Exception as e:
        return f"Error storing memory: {str(e)}"


@tool
def list_recent_memories(limit: int = 5) -> str:
    """
    List the most recent memories stored.
    
    Args:
        limit: Number of memories to retrieve (default: 5)
    
    Returns:
        List of recent memories
    """
    try:
        table = dynamodb_resource.Table(MEMORY_TABLE_NAME)
        
        response = table.scan(Limit=limit)
        items = response.get('Items', [])
        
        if not items:
            return "No memories found"
        
        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        result = f"Recent memories ({len(items)}):\n\n"
        for i, item in enumerate(items[:limit], 1):
            result += f"{i}. [{item['memory_id']}]\n"
            result += f"   TLDR: {item.get('tldr', 'N/A')}\n"
            result += f"   Hashtags: {', '.join(item.get('hashtags', []))}\n"
            result += f"   Created: {item.get('created_at', 'N/A')}\n"
            if 'source_url' in item:
                result += f"   Source: {item['source_url']}\n"
            result += "\n"
        
        return result.strip()
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            return f"Memory table '{MEMORY_TABLE_NAME}' does not exist yet"
        return f"Error listing memories: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_memories_by_hashtag(hashtag: str) -> str:
    """
    Search for memories containing a specific hashtag.
    
    Args:
        hashtag: The hashtag to search for (with or without #)
    
    Returns:
        List of matching memories
    """
    try:
        # Normalize hashtag
        if not hashtag.startswith('#'):
            hashtag = f"#{hashtag}"
        
        table = dynamodb_resource.Table(MEMORY_TABLE_NAME)
        response = table.scan()
        items = response.get('Items', [])
        
        # Filter items with matching hashtag
        matching_items = [
            item for item in items 
            if hashtag.lower() in [tag.lower() for tag in item.get('hashtags', [])]
        ]
        
        if not matching_items:
            return f"No memories found with hashtag: {hashtag}"
        
        result = f"Memories with {hashtag} ({len(matching_items)}):\n\n"
        for i, item in enumerate(matching_items, 1):
            result += f"{i}. [{item['memory_id']}]\n"
            result += f"   TLDR: {item.get('tldr', 'N/A')}\n"
            result += f"   Created: {item.get('created_at', 'N/A')}\n\n"
        
        return result.strip()
    
    except Exception as e:
        return f"Error searching memories: {str(e)}"


# ============================================================================
# Helper Functions
# ============================================================================

def _convert_floats_to_decimal(obj):
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimal(item) for item in obj]
    return obj


def extract_urls(text: str) -> list:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def create_memory_prompt(user_input: str, webpage_content: str = None) -> str:
    """Create a structured prompt for the agent."""
    prompt = f"User input: {user_input}\n\n"
    
    if webpage_content:
        prompt += f"Webpage content:\n{webpage_content}\n\n"
    
    prompt += """Please process this information and:
1. Create a concise TLDR (1-3 sentences)
2. Generate 3-5 relevant hashtags (format: #hashtag)
3. Store this as a memory item

Use the tools available to:
- First, ensure the memory table exists
- Then, store the memory with appropriate metadata"""
    
    return prompt


# ============================================================================
# Main Logic
# ============================================================================

def main(user_input):
    """Main function to process user input and store memories."""
    
    # Initialize agent with tools
    tools = [
        fetch_webpage_content,
        ensure_memory_table_exists,
        store_memory_item,
        list_recent_memories,
        search_memories_by_hashtag
    ]
    
    agent = Agent(
        system_prompt=SYSTEM_PROMPT,
        model=model,
        tools=tools,
        callback_handler=PrintingCallbackHandler()
    )
    
    # Check if input contains URLs
    urls = extract_urls(user_input)
    
    if urls:
        logger.info(f"Detected URL(s): {urls}")
        prompt = f"""The user wants to store this information: "{user_input}"

I found a URL: {urls[0]}

Please:
1. Use fetch_webpage_content to retrieve the webpage
2. Use ensure_memory_table_exists to make sure the table is ready
3. Analyze the content and create a TLDR (1-3 sentences) and 3-5 hashtags
4. Use store_memory_item to save everything

The content should include the original user input and the webpage content."""
    else:
        logger.info("No URLs detected, storing direct input")
        prompt = f"""The user wants to store this information: "{user_input}"

Please:
1. Use ensure_memory_table_exists to make sure the table is ready
2. Create a TLDR (1-3 sentences) and 3-5 hashtags from the input
3. Use store_memory_item to save the information"""
    
    # Run the agent
    result = agent(prompt)
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(result)


def get_stdin():
    """Get input from stdin if available."""
    if sys.stdin.isatty():
        return ''  # Interactive terminal - no piped input
    else:
        return sys.stdin.read().strip()  # Input is being piped or redirected


if __name__ == "__main__":
    user_input = get_stdin()
    if user_input:
        main(user_input)
    else:
        print("No input provided")
        print("\nUsage:")
        print('  echo "Store this web page: https://example.com" | python memory_agent.py')
        print('  echo "Remember that Python 3.12 was released in October 2023" | python memory_agent.py')
        sys.exit(1)
