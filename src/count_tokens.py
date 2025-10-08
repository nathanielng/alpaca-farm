#!/usr/bin/env python

"""
Token Counting Program

This script counts tokens using either tiktoken 
or an approximate word-based method as fallback.

Features:
- Supports both tiktoken and approximate word-based token counting
- Supports reading input from stdin when used in pipes

Usage:
    echo "text" | python count_tokens.py
    cat filename.txt | python count_tokens.py

Output:
    <number_of_tokens>

Requirements:
    - tiktoken (optional) for accurate token counting
"""

import re
import sys


def count_tokens_approximate(text):
    """
    Approximate token count based on word count.
    This is a rough approximation - about 0.75 words per token.
    
    Args:
        text (str): The text to count tokens for
    
    Returns:
        int: The approximate number of tokens
    """
    words = re.findall(r'\b\w+\b', text)
    return int(len(words) * 1.33)


def count_tokens_tiktoken(text, model="cl100k_base"):
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        model (str): The encoding to use (default: cl100k_base which is used by many models)
    
    Returns:
        int: The number of tokens
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(model)
        tokens = encoding.encode(text)
        return len(tokens)

    except Exception as e:
        print(f"Error counting tokens with tiktoken: {e}")
        # Fallback to approximate method
        return count_tokens_approximate(text)


if __name__ == "__main__":
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
    if sys.stdin.isatty():
        # Interactive terminal - no piped input, run interactive agent
        pass
    else:
        # Input is being piped or redirected
        user_input = sys.stdin.read().strip()
        if user_input:
            tokens = count_tokens_tiktoken(user_input)
            print(tokens)
        else:
            print("No input provided")
