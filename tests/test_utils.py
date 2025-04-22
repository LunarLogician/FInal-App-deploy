import pytest
from utils import is_english, get_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@pytest.fixture
def model_and_tokenizer():
    # Load a small model for testing
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def test_is_english():
    assert is_english("This is an English text.") is True
    assert is_english("Ceci est un texte en français.") is False
    assert is_english("これは日本語のテキストです。") is False
    assert is_english("") is False
    assert is_english("12345") is False

def test_get_score(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "This is a test text for scoring."
    
    score = get_score(text, model, tokenizer)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_get_score_empty_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    with pytest.raises(ValueError):
        get_score("", model, tokenizer)

def test_get_score_invalid_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    with pytest.raises(Exception):
        get_score(None, model, tokenizer) 