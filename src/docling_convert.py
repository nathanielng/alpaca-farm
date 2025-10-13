#!/usr/bin/env python3

"""
Docling Converter

A utility for converting various document formats (PDF, DOCX, HTML, etc.) to markdown
using docling

Usage:
  python docling_convert.py filename.pdf
  echo "https://arxiv.org/pdf/2206.01062" | docling_convert.py

  
Installation:
  uv pip install docling
"""

import argparse
import logging
import os
import subprocess
import sys

from docling.document_converter import DocumentConverter as DoclingConverter
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.document_converter import PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from pathlib import Path
from urllib.parse import urlparse


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("strands").setLevel(logging.INFO)


def convert_with_docling(file_path_or_url: str) -> str:
    """
    Convert document to markdown using docling and return the markdown text.

    Args:
        file_path_or_url (str): Path to local file or URL of document to convert

    Returns:
        str: Converted markdown text if successful, '' if conversion fails

    Reference:
        https://github.com/docling-project/docling
    """
    try:
        logger.info(f"Converting {file_path_or_url} to markdown using docling...")
        
        converter = DoclingConverter()
        result = converter.convert(file_path_or_url)
        
        markdown_str = result.document.export_to_markdown()
        logger.info(f"Successfully converted {file_path_or_url} to markdown with docling")
        return markdown_str

    except Exception as e:
        logger.error(f"Error converting to markdown with docling: {e}")
        return ''


def convert_with_docling_cli(file_path_or_url: str):
    """
    Call docling CLI to convert document to markdown
    
    Args:
        file_path_or_url (str): Path to local file or URL of document to convert
            e.g a URL to an arXiv paper
        
    Returns:
        str: Command output if successful, empty string if failed

    Examples:
        docling https://arxiv.org/pdf/2206.01062
        docling --pipeline vlm --vlm-model granite_docling https://arxiv.org/pdf/2206.01062

    Reference:
        https://docling-project.github.io/docling/usage/
    """

    try:
        # Run docling command and capture output
        result = subprocess.run(
            ["docling", file_path_or_url],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docling command: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error running docling command: {e}")
        return ""    


def main(user_input, use_cli = True):
    if use_cli:
        print(convert_with_docling_cli(user_input))
    else:
        print(convert_with_docling(user_input))

def get_stdin():
    # Check if stdin is connected to a terminal (interactive) or a pipe/file
    if sys.stdin.isatty():
        return ''  # Interactive terminal - no piped input
    else:
        return sys.stdin.read().strip()  # Input is being piped or redirected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert document to markdown using docling")
    parser.add_argument('--file_path_or_url', default='')
    parser.add_argument('--use_cli', action='store_true')
    args = parser.parse_args()

    user_input = get_stdin()
    if user_input:
        logger.info(f'Using docling to parser: {user_input}, with --use_cli={args.use_cli}')
        main(user_input, args.use_cli)
    else:
        file_path_or_url = args.file_path_or_url
        if file_path_or_url:
            markdown_str = convert_with_docling_cli(file_path_or_url)
            if markdown_str:
                print(markdown_str)
            else:
                print(f"Failed to convert {file_path_or_url} to markdown")
