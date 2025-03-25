"""
Parser for Remedy ticket data.
"""
import logging
import re
import nltk
from datetime import datetime

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

def parse_ticket_description(description):
    """Parse ticket description to extract structured information.
    
    Args:
        description: Ticket description text
        
    Returns:
        dict: Dictionary with parsed information
    """
    if not description:
        return {"text": "", "sections": {}}
    
    # Split description into sentences
    sentences = nltk.sent_tokenize(description)
    
    # Extract key sections using regex patterns
    problem_section = extract_section(description, ["problem:", "issue:", "error:"])
    action_section = extract_section(description, ["action taken:", "steps taken:", "troubleshooting:"])
    resolution_section = extract_section(description, ["resolution:", "solution:", "workaround:"])
    
    # Extract error codes if present
    error_codes = extract_error_codes(description)
    
    # Process timestamps
    timestamps = extract_timestamps(description)
    
    return {
        "text": description,
        "sentences": sentences,
        "sections": {
            "problem": problem_section,
            "action": action_section,
            "resolution": resolution_section
        },
        "error_codes": error_codes,
        "timestamps": timestamps
    }

def extract_section(text, section_markers):
    """Extract a section from text using markers.
    
    Args:
        text: The text to extract from
        section_markers: List of possible section marker strings
        
    Returns:
        str: Extracted section text
    """
    for marker in section_markers:
        pattern = re.compile(rf"{marker}\s*(.*?)(?=\n\n|\n[A-Za-z]+:|\Z)", re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_error_codes(text):
    """Extract error codes from text.
    
    Args:
        text: Text to extract from
        
    Returns:
        list: List of error codes
    """
    # Common error code patterns
    patterns = [
        r"error\s+code[s]?[\s:]*([\w\d-]+)",  # Error code: ABC123
        r"error[\s:]*([\w\d-]{3,})",          # Error: ABC123
        r"exception[\s:]*([\w\d.-]{3,})",     # Exception: System.NullReference
        r"status code[\s:]*([\d]{3})",        # Status code: 404
        r"\b([A-Z]{2,}-\d{3,})\b"             # ISSUE-1234 format
    ]
    
    error_codes = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            error_codes.append(match.group(1))
    
    return error_codes

def extract_timestamps(text):
    """Extract timestamps from text.
    
    Args:
        text: Text to extract from
        
    Returns:
        list: List of timestamp strings
    """
    # Common timestamp patterns
    patterns = [
        r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}:\d{2}",  # MM/DD/YYYY HH:MM:SS
        r"\d{1,2}-\d{1,2}-\d{2,4}\s+\d{1,2}:\d{2}:\d{2}",  # MM-DD-YYYY HH:MM:SS
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",           # ISO format
        r"\d{1,2}/\d{1,2}/\d{2,4}",                       # MM/DD/YYYY
        r"\d{1,2}-\d{1,2}-\d{2,4}"                        # MM-DD-YYYY
    ]
    
    timestamps = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            timestamps.append(match.group(0))
    
    return timestamps

def format_ticket_for_display(ticket):
    """Format ticket data for display.
    
    Args:
        ticket: Ticket data dictionary
        
    Returns:
        str: Formatted ticket text
    """
    if not ticket:
        return ""
    
    # Parse the description
    parsed_description = parse_ticket_description(ticket.get('description', ''))
    
    # Format the ticket
    sections = []
    
    # Ticket header
    sections.append(f"Ticket ID: {ticket.get('id', 'N/A')}")
    sections.append(f"Status: {ticket.get('status', 'N/A')}")
    
    # Add submitter and assignee
    if ticket.get('submitter'):
        sections.append(f"Submitted by: {ticket.get('submitter')}")
    if ticket.get('assignee'):
        sections.append(f"Assigned to: {ticket.get('assignee')}")
    
    # Add priority and impact if available
    if ticket.get('priority') or ticket.get('impact'):
        priority_impact = []
        if ticket.get('priority'):
            priority_impact.append(f"Priority: {ticket.get('priority')}")
        if ticket.get('impact'):
            priority_impact.append(f"Impact: {ticket.get('impact')}")
        sections.append(" | ".join(priority_impact))
    
    # Add problem description
    if parsed_description['sections']['problem']:
        sections.append("\nProblem:")
        sections.append(parsed_description['sections']['problem'])
    elif parsed_description['text']:
        sections.append("\nDescription:")
        sections.append(parsed_description['text'][:500] + ('...' if len(parsed_description['text']) > 500 else ''))
    
    # Add action taken if available
    if parsed_description['sections']['action']:
        sections.append("\nAction Taken:")
        sections.append(parsed_description['sections']['action'])
    
    # Add resolution if available
    if parsed_description['sections']['resolution']:
        sections.append("\nResolution:")
        sections.append(parsed_description['sections']['resolution'])
    
    # Add error codes if found
    if parsed_description['error_codes']:
        sections.append("\nError Codes:")
        sections.append(", ".join(parsed_description['error_codes']))
    
    return "\n".join(sections)