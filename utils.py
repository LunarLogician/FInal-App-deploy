import torch
from langdetect import detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_english(text: str) -> bool:
    """
    Check if text is English using langdetect.
    Returns False for empty strings, numbers, or non-English text.
    """
    if not text or text.isspace():
        return False
    
    # Check if text is just numbers or special characters
    if text.replace('.', '').replace(',', '').replace(' ', '').isdigit():
        return False
    
    try:
        return detect(text) == 'en'
    except:
        return False

def get_score(text: str, model, tokenizer) -> float:
    """
    Get model score for text.
    Raises ValueError for empty text and Exception for invalid text.
    """
    if not text or text.isspace():
        raise ValueError("Text cannot be empty or whitespace")
    
    try:
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            return float(scores[0][1])  # Return probability of positive class
    except Exception as e:
        raise Exception(f"Error processing text: {str(e)}")