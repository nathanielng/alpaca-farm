#!/usr/bin/env python

"""
AWS Knowledge MCP Server - Unix Pipe Format
============================================

A command line utility that processes AWS-related questions from stdin using 
the AWS Knowledge MCP Server and Strands Agent.

The AWS Knowledge Model Context Protocol (MCP) Server is a fully managed remote 
MCP server that surfaces authoritative AWS knowledge in an LLM-compatible format, 
including documentation, blog posts, What's New announcements, Well-Architected 
best practices, code samples, and other official AWS content.

AWS Knowledge MCP Server enables clients and foundation models (FMs) that support 
MCP to ground their responses in trusted AWS context, guidance, and best practices, 
providing the guidance needed for accurate reasoning and consistent execution, 
while reducing manual context management. Customers can now focus on business 
problems instead of searching for information manually.

The server is publicly accessible at no cost and does not require an AWS account. 
Usage is subject to rate limits. Give your developers and agents access to the 
most up-to-date AWS information today by configuring your MCP clients to use the 
AWS Knowledge MCP Server endpoint, and follow the Getting Started guide for setup 
instructions.

Important Note: Not all MCP clients today support remote servers. Please make sure 
that your client supports remote MCP servers or that you have a suitable proxy 
setup to use this server.

Key Features
- Real-time access to AWS documentation, API references, and architectural guidance
- Less local setup compared to client-hosted servers
- Structured access to AWS knowledge for AI agents

AWS Knowledge capabilities
- Best practices: Discover best practices around using AWS APIs and services
- API documentation: Learn about how to call APIs including required and optional parameters and flags
- Getting started: Find out how to quickly get started using AWS services while following best practices
- The latest information: Access the latest announcements about new AWS services and features

Tools
- search_documentation: Search across all AWS documentation
- read_documentation: Retrieve and convert AWS documentation pages to markdown
- recommend: Get content recommendations for AWS documentation pages

Current knowledge sources
- The latest AWS docs
- API references
- What's New posts
- Getting Started information
- Builder Center
- Blog posts
- Architectural references
- Well-Architected guidance

FAQs
1. Should I use the local AWS Documentation MCP Server or the remote AWS Knowledge MCP Server?
   The Knowledge server indexes a variety of information sources in addition to AWS Documentation 
   including What's New Posts, Getting Started Information, guidance from the Builder Center, Blog 
   posts, Architectural references, and Well-Architected guidance. If your MCP client supports 
   remote servers you can easily try the Knowledge MCP server to see if it suits your needs.
2. Do I need network access to use the AWS Knowledge MCP Server?
   Yes, you will need to be able to access the public internet to access the AWS Knowledge MCP Server.
3. Do I need an AWS account?
   No. You can get started with the Knowledge MCP server without an AWS account. The Knowledge MCP 
   is subject to the AWS Site Terms

Example usage:
    echo "What are the best practices for securing an S3 bucket?" | python strands_aws_knowledge.py
    echo "How do I enable multi-factor authentication for IAM users?" | python strands_aws_knowledge.py
    echo "Explain the difference between AWS Lambda and AWS Fargate" | python strands_aws_knowledge.py
    echo "What's the recommended architecture for a highly available web application?" | python strands_aws_knowledge.py
    echo "Show me recent announcements about Amazon EKS" | python strands_aws_knowledge.py
    cat questions.txt | python strands_aws_knowledge.py
    python strands_aws_knowledge.py < input.txt
    python strands_aws_knowledge.py --npx < questions.txt

Announcement: https://aws.amazon.com/about-aws/whats-new/2025/07/aws-knowledge-mcp-server-available-preview/
Github: https://github.com/awslabs/mcp/tree/main/src/aws-knowledge-mcp-server
"""

import argparse
import logging
import os
import sys

from botocore.config import Config
from mcp import stdio_client, StdioServerParameters
from shutil import which
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from strands.handlers.callback_handler import PrintingCallbackHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logging.getLogger("strands").setLevel(logging.INFO)

# Configuration with environment variable fallbacks
HOME = os.getenv('HOME')
BEDROCK_REGION = os.getenv("BEDROCK_REGION", 'us-west-2')
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

def create_mcp_client(use_npx: bool = False) -> MCPClient:
    """Create an MCP client for the AWS Knowledge MCP Server.
    
    Args:
        use_npx: If True, use npx instead of uvx for the MCP client
        
    Returns:
        MCPClient: Configured MCP client
        
    Raises:
        RuntimeError: If required command is not found
    """
    if use_npx:
        cmd = which('npx')
        if not cmd:
            raise RuntimeError("npx command not found. Please install Node.js and npm.")
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=cmd,
                args=[
                    'mcp-remote',
                    'https://knowledge-mcp.global.api.aws'
                ]
            )
        ))
    else:
        cmd = which('uvx')
        if not cmd:
            raise RuntimeError("uvx command not found. Please install uvx.")
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=cmd,
                args=[
                    'mcp-proxy',
                    '--transport',
                    'streamablehttp',
                    'https://knowledge-mcp.global.api.aws'
                ]
            )
        ))

model = BedrockModel(
    model_id=BEDROCK_MODEL_ID,
    max_tokens=2048,
    boto_client_config=Config(
        region_name=BEDROCK_REGION,
        read_timeout=120,
        connect_timeout=120,
        retries=dict(max_attempts=3, mode="adaptive"),
    ),
    temperature=0.1
)

SYSTEM_PROMPT = """You are an AWS Knowledge assistant with access to AWS documentation and guidance.

Use the available tools to:
- Search AWS documentation and best practices
- Access API references and getting started guides
- Find architectural guidance and Well-Architected best practices
- Stay up to date with AWS announcements and blog posts

Provide accurate AWS knowledge and guidance based on user questions.
"""

def process_input(input_text, use_npx=False):
    """
    Process AWS-related questions using Strands Agent with AWS Knowledge MCP Server

    Args:
        input_text (str): The AWS-related question to process
        use_npx (bool): If True, use npx instead of uvx for the MCP client

    Returns:
        str: output generated by the agent
    """
    try:
        # Create MCP client
        mcp_client = create_mcp_client(use_npx=use_npx)
        
        with mcp_client:
            # Get available tools
            tools = mcp_client.list_tools_sync()
            
            # Create agent
            aws_knowledge_agent = Agent(
                system_prompt=SYSTEM_PROMPT,
                model=model,
                tools=tools,
                callback_handler=PrintingCallbackHandler()
            )
            
            return aws_knowledge_agent(input_text)
            
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        sys.exit(1)

def main(stdin_input, args):
    print(process_input(stdin_input, use_npx=args.npx))

def get_stdin():
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
    if sys.stdin.isatty():
        return ''  # Interactive terminal - no piped input
    else:
        return sys.stdin.read().strip()  # Input is being piped or redirected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWS Knowledge MCP Server - Process AWS questions from stdin"
    )
    parser.add_argument(
        '--npx',
        action='store_true',
        help='Use npx instead of uvx for the MCP client'
    )
    args = parser.parse_args()

    stdin_input = get_stdin()
    if stdin_input:
        main(stdin_input, args)
    else:
        print("No input provided. Usage: echo 'Your AWS question' | python strands_aws_knowledge.py")
