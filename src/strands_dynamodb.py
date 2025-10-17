#!/usr/bin/env python

import logging
import os
import sys
from decimal import Decimal

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.models.bedrock import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient

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
DYNAMODB_REGION = os.getenv("DYNAMODB_REGION", BEDROCK_REGION)

# Initialize AWS clients
dynamodb_client = boto3.client('dynamodb', region_name=DYNAMODB_REGION)
dynamodb_resource = boto3.resource('dynamodb', region_name=DYNAMODB_REGION)

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

SYSTEM_PROMPT = """You are a helpful AI assistant with access to DynamoDB operations. 
You can create, read, update, and delete items in DynamoDB tables, as well as manage the tables themselves."""

# ============================================================================
# DynamoDB Table Management Tools
# ============================================================================

@tool
def create_dynamodb_table(
    table_name: str,
    partition_key: str,
    partition_key_type: str = "S",
    sort_key: str = None,
    sort_key_type: str = "S",
    read_capacity: int = 5,
    write_capacity: int = 5
) -> str:
    """
    Create a new DynamoDB table.
    
    Args:
        table_name: Name of the table to create
        partition_key: Name of the partition key attribute
        partition_key_type: Type of partition key (S=String, N=Number, B=Binary)
        sort_key: Optional name of the sort key attribute
        sort_key_type: Type of sort key (S=String, N=Number, B=Binary)
        read_capacity: Read capacity units (default: 5)
        write_capacity: Write capacity units (default: 5)
    
    Returns:
        Success or error message
    """
    try:
        # Build key schema
        key_schema = [
            {'AttributeName': partition_key, 'KeyType': 'HASH'}
        ]
        
        attribute_definitions = [
            {'AttributeName': partition_key, 'AttributeType': partition_key_type}
        ]
        
        if sort_key:
            key_schema.append({'AttributeName': sort_key, 'KeyType': 'RANGE'})
            attribute_definitions.append({'AttributeName': sort_key, 'AttributeType': sort_key_type})
        
        # Create table
        response = dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
            BillingMode='PROVISIONED',
            ProvisionedThroughput={
                'ReadCapacityUnits': read_capacity,
                'WriteCapacityUnits': write_capacity
            }
        )
        
        logger.info(f"Creating table {table_name}...")
        return f"Table '{table_name}' is being created. Status: {response['TableDescription']['TableStatus']}"
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceInUseException':
            return f"Error: Table '{table_name}' already exists"
        else:
            return f"Error creating table: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def list_dynamodb_tables() -> str:
    """
    List all DynamoDB tables in the region.
    
    Returns:
        List of table names or error message
    """
    try:
        response = dynamodb_client.list_tables()
        tables = response.get('TableNames', [])
        
        if not tables:
            return "No DynamoDB tables found in this region"
        
        return f"DynamoDB tables ({len(tables)}):\n" + "\n".join(f"  - {table}" for table in tables)
    
    except ClientError as e:
        return f"Error listing tables: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def delete_dynamodb_table(table_name: str) -> str:
    """
    Delete a DynamoDB table.
    
    Args:
        table_name: Name of the table to delete
    
    Returns:
        Success or error message
    """
    try:
        dynamodb_client.delete_table(TableName=table_name)
        logger.info(f"Deleting table {table_name}...")
        return f"Table '{table_name}' is being deleted"
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            return f"Error: Table '{table_name}' does not exist"
        else:
            return f"Error deleting table: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def describe_dynamodb_table(table_name: str) -> str:
    """
    Get detailed information about a DynamoDB table.
    
    Args:
        table_name: Name of the table to describe
    
    Returns:
        Table details or error message
    """
    try:
        response = dynamodb_client.describe_table(TableName=table_name)
        table = response['Table']
        
        result = f"Table: {table['TableName']}\n"
        result += f"Status: {table['TableStatus']}\n"
        result += f"Item Count: {table.get('ItemCount', 'N/A')}\n"
        result += f"Size: {table.get('TableSizeBytes', 0)} bytes\n"
        result += f"Creation Time: {table.get('CreationDateTime', 'N/A')}\n"
        result += "\nKey Schema:\n"
        for key in table['KeySchema']:
            result += f"  - {key['AttributeName']} ({key['KeyType']})\n"
        
        return result
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            return f"Error: Table '{table_name}' does not exist"
        else:
            return f"Error describing table: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ============================================================================
# DynamoDB Item CRUD Operations
# ============================================================================

@tool
def put_dynamodb_item(table_name: str, item: dict) -> str:
    """
    Create or update an item in a DynamoDB table.
    
    Args:
        table_name: Name of the table
        item: Dictionary containing the item data (must include primary key)
    
    Returns:
        Success or error message
    """
    try:
        table = dynamodb_resource.Table(table_name)
        
        # Convert float to Decimal for DynamoDB compatibility
        item = _convert_floats_to_decimal(item)
        
        table.put_item(Item=item)
        logger.info(f"Item added to table {table_name}")
        return f"Successfully added/updated item in table '{table_name}'"
    
    except ClientError as e:
        return f"Error putting item: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def get_dynamodb_item(table_name: str, key: dict) -> str:
    """
    Retrieve an item from a DynamoDB table.
    
    Args:
        table_name: Name of the table
        key: Dictionary containing the primary key (e.g., {'id': '123'})
    
    Returns:
        Item data or error message
    """
    try:
        table = dynamodb_resource.Table(table_name)
        response = table.get_item(Key=key)
        
        if 'Item' in response:
            item = _convert_decimal_to_float(response['Item'])
            return f"Item found:\n{_format_item(item)}"
        else:
            return f"No item found with key: {key}"
    
    except ClientError as e:
        return f"Error getting item: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def update_dynamodb_item(
    table_name: str,
    key: dict,
    updates: dict
) -> str:
    """
    Update specific attributes of an item in a DynamoDB table.
    
    Args:
        table_name: Name of the table
        key: Dictionary containing the primary key
        updates: Dictionary of attributes to update (e.g., {'status': 'active', 'count': 5})
    
    Returns:
        Success or error message
    """
    try:
        table = dynamodb_resource.Table(table_name)
        
        # Convert float to Decimal
        updates = _convert_floats_to_decimal(updates)
        
        # Build update expression
        update_expression = "SET " + ", ".join(f"#{k} = :{k}" for k in updates.keys())
        expression_attribute_names = {f"#{k}": k for k in updates.keys()}
        expression_attribute_values = {f":{k}": v for k, v in updates.items()}
        
        table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )
        
        logger.info(f"Item updated in table {table_name}")
        return f"Successfully updated item in table '{table_name}'"
    
    except ClientError as e:
        return f"Error updating item: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def delete_dynamodb_item(table_name: str, key: dict) -> str:
    """
    Delete an item from a DynamoDB table.
    
    Args:
        table_name: Name of the table
        key: Dictionary containing the primary key
    
    Returns:
        Success or error message
    """
    try:
        table = dynamodb_resource.Table(table_name)
        table.delete_item(Key=key)
        logger.info(f"Item deleted from table {table_name}")
        return f"Successfully deleted item from table '{table_name}'"
    
    except ClientError as e:
        return f"Error deleting item: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def scan_dynamodb_table(
    table_name: str,
    limit: int = 10,
    filter_expression: str = None
) -> str:
    """
    Scan a DynamoDB table and return items (use sparingly, expensive operation).
    
    Args:
        table_name: Name of the table
        limit: Maximum number of items to return (default: 10)
        filter_expression: Optional filter expression (e.g., "attribute_exists(email)")
    
    Returns:
        List of items or error message
    """
    try:
        table = dynamodb_resource.Table(table_name)
        
        scan_kwargs = {'Limit': limit}
        if filter_expression:
            scan_kwargs['FilterExpression'] = filter_expression
        
        response = table.scan(**scan_kwargs)
        items = response.get('Items', [])
        
        if not items:
            return f"No items found in table '{table_name}'"
        
        items = [_convert_decimal_to_float(item) for item in items]
        result = f"Found {len(items)} item(s) in '{table_name}':\n\n"
        for i, item in enumerate(items, 1):
            result += f"Item {i}:\n{_format_item(item)}\n\n"
        
        return result.strip()
    
    except ClientError as e:
        return f"Error scanning table: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def query_dynamodb_table(
    table_name: str,
    partition_key_name: str,
    partition_key_value: str,
    limit: int = 10
) -> str:
    """
    Query a DynamoDB table by partition key.
    
    Args:
        table_name: Name of the table
        partition_key_name: Name of the partition key attribute
        partition_key_value: Value to query for
        limit: Maximum number of items to return (default: 10)
    
    Returns:
        List of items or error message
    """
    try:
        table = dynamodb_resource.Table(table_name)
        
        from boto3.dynamodb.conditions import Key
        
        response = table.query(
            KeyConditionExpression=Key(partition_key_name).eq(partition_key_value),
            Limit=limit
        )
        
        items = response.get('Items', [])
        
        if not items:
            return f"No items found with {partition_key_name} = '{partition_key_value}'"
        
        items = [_convert_decimal_to_float(item) for item in items]
        result = f"Found {len(items)} item(s):\n\n"
        for i, item in enumerate(items, 1):
            result += f"Item {i}:\n{_format_item(item)}\n\n"
        
        return result.strip()
    
    except ClientError as e:
        return f"Error querying table: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


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


def _convert_decimal_to_float(obj):
    """Convert Decimal values to float for display."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimal_to_float(item) for item in obj]
    return obj


def _format_item(item):
    """Format item dictionary for readable output."""
    result = []
    for key, value in item.items():
        result.append(f"  {key}: {value}")
    return "\n".join(result)


# ============================================================================
# Agent Setup
# ============================================================================

# Collect all tools
tools = [
    # Table management
    create_dynamodb_table,
    list_dynamodb_tables,
    delete_dynamodb_table,
    describe_dynamodb_table,
    # Item operations
    put_dynamodb_item,
    get_dynamodb_item,
    update_dynamodb_item,
    delete_dynamodb_item,
    scan_dynamodb_table,
    query_dynamodb_table,
]

prompts = [
    "Create a table called 'users' with partition key 'user_id'",
    "List all DynamoDB tables",
    "Add a user with user_id='123' and name='John Doe' to the users table"
]

def main(user_input):
    aws_agent = Agent(
        system_prompt = SYSTEM_PROMPT,
        model = model,
        tools = tools,
        callback_handler = PrintingCallbackHandler()
    )
    print(aws_agent(user_input))

def get_stdin():
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
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
