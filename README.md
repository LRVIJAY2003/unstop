"""
Parser for Confluence content.
"""
import re
import logging
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

def parse_confluence_content(html_content):
    """Parse Confluence HTML content preserving structure.
    
    Args:
        html_content: HTML content from Confluence
        
    Returns:
        str: Parsed content with structure preserved
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Process tables
    tables = soup.find_all('table')
    for table in tables:
        try:
            # Extract table data
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cells.append(td.get_text(strip=True))
                rows.append(cells)
            
            # Convert to DataFrame and then to string
            if rows:
                df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
                table_str = df.to_string(index=False)
                # Create a new tag with the table text
                table_text = soup.new_tag('pre')
                table_text.string = f"\nTABLE:\n{table_str}\n"
                table.replace_with(table_text)
        except Exception as e:
            logger.warning(f"Error parsing table: {str(e)}")
    
    # Process code blocks
    code_blocks = soup.find_all('ac:structured-macro', {'ac:name': 'code'})
    for code_block in code_blocks:
        try:
            code_content = code_block.find('ac:plain-text-body')
            if code_content:
                code_text = code_content.get_text()
                # Create a new tag with the code
                formatted_code = soup.new_tag('pre')
                formatted_code.string = f"\nCODE:\n{code_text}\n"
                code_block.replace_with(formatted_code)
        except Exception as e:
            logger.warning(f"Error parsing code block: {str(e)}")
    
    # Process lists
    lists = soup.find_all(['ul', 'ol'])
    for list_elem in lists:
        try:
            list_items = []
            for idx, li in enumerate(list_elem.find_all('li')):
                prefix = "• " if list_elem.name == 'ul' else f"{idx+1}. "
                list_items.append(f"{prefix}{li.get_text(strip=True)}")
            
            # Create a new tag with the list text
            list_text = soup.new_tag('div')
            list_text.string = "\n".join(list_items)
            list_elem.replace_with(list_text)
        except Exception as e:
            logger.warning(f"Error parsing list: {str(e)}")
    
    # Process images
    images = soup.find_all('ac:image')
    for image in images:
        try:
            # Just add a placeholder for images
            image_placeholder = soup.new_tag('p')
            image_placeholder.string = "[IMAGE]"
            image.replace_with(image_placeholder)
        except Exception as e:
            logger.warning(f"Error processing image: {str(e)}")
    
    # Get the text content
    content = soup.get_text(separator="\n")
    
    # Clean up excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content


def extract_tables_from_html(html_content):
    """Extract tables from HTML content.
    
    Args:
        html_content: HTML content
        
    Returns:
        list: List of pandas DataFrames
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []
    
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(cells)
        
        if rows:
            # Use first row as header
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 0 else None)
            tables.append(df)
    
    return tables


def extract_code_blocks(html_content):
    """Extract code blocks from HTML content.
    
    Args:
        html_content: HTML content
        
    Returns:
        list: List of code blocks
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    code_blocks = []
    
    # Find Confluence code macros
    for code_block in soup.find_all('ac:structured-macro', {'ac:name': 'code'}):
        try:
            code_content = code_block.find('ac:plain-text-body')
            if code_content:
                code_blocks.append(code_content.get_text())
        except Exception as e:
            logger.warning(f"Error extracting code block: {str(e)}")
    
    # Also find standard code elements
    for code in soup.find_all('code'):
        code_blocks.append(code.get_text())
    
    return code_blocks