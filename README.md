"""
Utilities for extracting data from Confluence content.
"""
import logging
import re
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)

def extract_structured_data(html_content):
    """Extract structured data from Confluence HTML.
    
    Args:
        html_content: HTML content from Confluence
        
    Returns:
        dict: Dictionary with extracted data
    """
    if not html_content:
        return {}
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract headings and their content
    headings_data = extract_headings_with_content(soup)
    
    # Extract tables
    tables_data = extract_tables(soup)
    
    # Extract lists
    lists_data = extract_lists(soup)
    
    # Extract metadata
    metadata = extract_metadata(soup)
    
    return {
        'headings': headings_data,
        'tables': tables_data,
        'lists': lists_data,
        'metadata': metadata
    }

def extract_headings_with_content(soup):
    """Extract headings and their content.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        list: List of dictionaries with heading and content
    """
    headings_data = []
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    for heading in headings:
        heading_text = heading.get_text(strip=True)
        level = int(heading.name[1])  # Extract heading level (h1 -> 1, etc.)
        
        # Get content until next heading of same or higher level
        content_elements = []
        next_element = heading.next_sibling
        
        while next_element:
            if next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if int(next_element.name[1]) <= level:
                    break
            
            if next_element.string and next_element.string.strip():
                content_elements.append(next_element.string.strip())
            elif next_element.get_text(strip=True):
                content_elements.append(next_element.get_text(strip=True))
                
            next_element = next_element.next_sibling
        
        content = ' '.join(content_elements)
        
        headings_data.append({
            'level': level,
            'text': heading_text,
            'content': content
        })
    
    return headings_data

def extract_tables(soup):
    """Extract tables from the soup.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        list: List of tables as pandas DataFrames converted to dictionaries
    """
    tables_data = []
    
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(cells)
        
        if rows:
            # Try to use first row as header
            if len(rows) > 1:
                df = pd.DataFrame(rows[1:], columns=rows[0])
            else:
                df = pd.DataFrame(rows)
                
            # Convert to dict for serialization
            tables_data.append(df.to_dict(orient='records'))
    
    return tables_data

def extract_lists(soup):
    """Extract lists from the soup.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        list: List of dictionaries with list information
    """
    lists_data = []
    
    for list_elem in soup.find_all(['ul', 'ol']):
        list_type = 'bullet' if list_elem.name == 'ul' else 'numbered'
        items = [li.get_text(strip=True) for li in list_elem.find_all('li')]
        
        lists_data.append({
            'type': list_type,
            'items': items
        })
    
    return lists_data

def extract_metadata(soup):
    """Extract metadata from the soup.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        dict: Dictionary with metadata
    """
    metadata = {}
    
    # Try to extract metadata from Confluence specific elements
    meta_elements = soup.find_all('meta')
    for meta in meta_elements:
        name = meta.get('name', '')
        content = meta.get('content', '')
        if name and content:
            metadata[name] = content
    
    # Extract page title
    title = soup.find('title')
    if title:
        metadata['title'] = title.get_text(strip=True)
    
    return metadata

def extract_code_samples(html_content):
    """Extract code samples from HTML content.
    
    Args:
        html_content: HTML content
        
    Returns:
        list: List of code samples
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    code_samples = []
    
    # Confluence code macro
    for code_macro in soup.find_all('ac:structured-macro', {'ac:name': 'code'}):
        code = code_macro.find('ac:plain-text-body')
        if code:
            language_param = code_macro.find('ac:parameter', {'ac:name': 'language'})
            language = language_param.get_text() if language_param else 'text'
            
            code_samples.append({
                'code': code.get_text(),
                'language': language
            })
    
    # Standard HTML code elements
    for code in soup.find_all('code'):
        parent = code.find_parent('pre')
        language = parent.get('class', [''])[0].replace('language-', '') if parent and parent.get('class') else 'text'
        
        code_samples.append({
            'code': code.get_text(),
            'language': language
        })
    
    return code_samples