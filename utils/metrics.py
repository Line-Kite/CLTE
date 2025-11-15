import re
import json


def extract_knowledge_from_text(text):
    """
    Extract knowledge content from model response text.
    
    Args:
        text: Model response text containing knowledge information
    
    Returns:
        Extracted knowledge string
    """
    # Clean text if it starts/ends with code block markers
    cleaned_text = text
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.lstrip("```json")
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text.rstrip("```")
    
    try:
        # Try to parse as JSON first
        knowledge_data = json.loads(cleaned_text)
        return knowledge_data.get("knowledge", "")
    except Exception:
        
        # Fallback extraction methods
        knowledge = ""
        if "\"knowledge\":" in cleaned_text:
            knowledge = cleaned_text.split("\"knowledge\":")[-1].strip()
        
        # Clean quotation marks
        if knowledge.startswith("\""):
            knowledge = knowledge.lstrip("\"")
        if knowledge.endswith("\""):
            knowledge = knowledge.rstrip("\"")
        
        return knowledge


def extract_last_knowledge_object(text):
    """
    Extract the last knowledge object from model response text using multiple pattern matching strategies.
    
    Args:
        text: Model response text
    
    Returns:
        Extracted knowledge string
    """
    # Pattern 1: Match ```json\n{...}\n``` format
    pattern_codeblock = r'```json\s*({.*?"knowledge":.*?})\s*```'
    
    # Pattern 2: Match multi-line JSON (without ```json markers)
    pattern_multiline = r'{\s*"knowledge":.*?}'
    
    # Pattern 3: Match inline {"knowledge": ...} format
    pattern_inline = r'{"knowledge":.*?}(?=(?:[^"]*"[^"]*")*[^"]*$)'

    # Try code block format first
    codeblock_matches = re.findall(pattern_codeblock, text, re.DOTALL)
    if codeblock_matches:
        return extract_knowledge_from_text(f"```json\n{codeblock_matches[-1]}\n```")

    # Try multi-line JSON format
    multiline_matches = re.findall(pattern_multiline, text, re.DOTALL)
    if multiline_matches:
        return extract_knowledge_from_text(multiline_matches[-1])

    # Try inline JSON format
    inline_matches = re.findall(pattern_inline, text)
    if inline_matches:
        return extract_knowledge_from_text(inline_matches[-1])

    return ""


def extract_last_boxed_answer(text):
    """
    Extract the answer option (A/B/C/D) from the last non-empty \\boxed{} command.
    
    Supports various formats including:
    - \\boxed{A}
    - \\boxed{\\text{A}}
    - \\boxed{A. answer text}
    - \\boxed{\\text{A. answer text}}
    
    Args:
        text (str): Input text containing \\boxed commands
        
    Returns:
        str or None: The extracted answer option (A, B, C, or D), 
                    or None if no valid option is found
    """
    # Regex pattern to match various \\boxed{} formats
    boxed_matches = re.findall(r'\\boxed\{(\s*\\text\{([^}]*)\}|\s*\{?([A-D][\.\s].*?\}?)\}?|\s*([A-D])\s*)\}', text)
    
    # Extract all matched options, prioritizing inner content
    extracted_options = []
    for match in boxed_matches:
        # Match format: \\boxed{\\text{...}}
        if match[1]:
            content = match[1]
        # Match format: \\boxed{A.xxx} or variants
        elif match[2]:
            content = match[2]
        # Match format: \\boxed{A}
        elif match[3]:
            content = match[3]
        else:
            continue
        
        # Remove extra braces and whitespace
        content = content.strip()
        if content:
            extracted_options.append(content)
    
    if not extracted_options:
        return None
    
    # Get the last boxed content
    last_boxed = extracted_options[-1]
    
    # Check if it's a single A/B/C/D option
    if re.fullmatch(r'[A-D]', last_boxed):
        return last_boxed
    
    # Check if it's A.xxx/B.xxx format
    match = re.match(r'^([A-D])[\.\s].*', last_boxed)
    if match:
        return match.group(1)
    
    return None


def calculate_accuracy(predictions, answers):
    """
    Calculate overall accuracy based on predictions and ground truth answers.
    
    Args:
        predictions (list): List of predicted answers (from extract_last_boxed_answer)
        answers (list): List of ground truth answers (A/B/C/D format)
        
    Returns:
        float: Overall accuracy score between 0.0 and 1.0
        
    Raises:
        ValueError: If the lengths of predictions and answers don't match
    """
    if len(predictions) != len(answers):
        raise ValueError("Number of predictions and answers must be equal")
    
    correct = 0
    for pred, ans in zip(predictions, answers):
        # Compare ignoring case, None predictions are considered incorrect
        if pred is not None and pred.upper() == ans.upper():
            correct += 1
    
    return correct / len(predictions) if len(predictions) > 0 else 0.0

