import json
import os
from typing import List, Dict, Any, Union

def save_json(data: List[Dict[str, Any]], file_path: str, indent: int = 4, ensure_ascii: bool = False) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: List of dictionaries to save as JSON
        file_path: Path where the JSON file will be created
        indent: Number of spaces for indentation (default: 4)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
        
    Raises:
        IOError: If there's an error writing to the file
        TypeError: If data is not JSON serializable
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, ensure_ascii=ensure_ascii)
    except (IOError, TypeError) as e:
        raise type(e)(f"Error writing to {file_path}: {str(e)}")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL (JSON Lines) file.
    
    Args:
        file_path: Path to the JSONL file to load
        
    Returns:
        List of dictionaries containing the parsed JSON objects
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If any line contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num} in file {file_path}: {e.msg}",
                    e.doc, e.pos
                )
    
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL (JSON Lines) file.
    
    Args:
        data: List of dictionaries to save as JSONL
        file_path: Path where the JSONL file will be created
        
    Raises:
        IOError: If there's an error writing to the file
        TypeError: If any item in data is not JSON serializable
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                file.write(json_line + '\n')
    except (IOError, TypeError) as e:
        raise type(e)(f"Error writing to {file_path}: {str(e)}")


def append_to_jsonl(item: Dict[str, Any], file_path: str) -> None:
    """
    Append a single item to a JSONL file.
    
    Args:
        item: Dictionary to append to the file
        file_path: Path to the JSONL file (will be created if it doesn't exist)
        
    Raises:
        IOError: If there's an error writing to the file
        TypeError: If the item is not JSON serializable
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')
    except (IOError, TypeError) as e:
        raise type(e)(f"Error appending to {file_path}: {str(e)}")


def batch_append_to_jsonl(items: List[Dict[str, Any]], file_path: str) -> None:
    """
    Append multiple items to a JSONL file efficiently.
    
    Args:
        items: List of dictionaries to append to the file
        file_path: Path to the JSONL file (will be created if it doesn't exist)
        
    Raises:
        IOError: If there's an error writing to the file
        TypeError: If any item is not JSON serializable
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            for item in items:
                json_line = json.dumps(item, ensure_ascii=False)
                file.write(json_line + '\n')
    except (IOError, TypeError) as e:
        raise type(e)(f"Error batch appending to {file_path}: {str(e)}")


def count_jsonl_lines(file_path: str) -> int:
    """
    Count the number of valid JSON lines in a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Number of valid JSON lines in the file
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Count non-empty lines
                count += 1
    return count
