import pytest
import httpx
import os
import tempfile
from main import app
from fastapi.testclient import TestClient
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def models_and_tokenizers(device):
    """Load models and tokenizers once for the entire test session"""
    # Load commitment model
    commitment_model = AutoModelForSequenceClassification.from_pretrained(
        "climatebert/distilroberta-base-climate-commitment"
    ).to(device)
    commitment_tokenizer = AutoTokenizer.from_pretrained(
        "climatebert/distilroberta-base-climate-commitment"
    )
    
    # Load specificity model
    specificity_model = AutoModelForSequenceClassification.from_pretrained(
        "climatebert/distilroberta-base-climate-specificity"
    ).to(device)
    specificity_tokenizer = AutoTokenizer.from_pretrained(
        "climatebert/distilroberta-base-climate-specificity"
    )
    
    # Load ESG model
    esg_model = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-esg-9-categories"
    ).to(device)
    esg_tokenizer = AutoTokenizer.from_pretrained(
        "yiyanghkust/finbert-esg-9-categories"
    )
    
    return {
        "commitment": (commitment_model, commitment_tokenizer),
        "specificity": (specificity_model, specificity_tokenizer),
        "esg": (esg_model, esg_tokenizer)
    }

@pytest.fixture
def client(models_and_tokenizers):
    """Create a test client for the FastAPI application"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def test_file():
    """Create a temporary test file"""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is a test file content.")
        f.flush()
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def chat_request_data():
    """Prepare chat request data"""
    return {
        "message": "What is the company's commitment to sustainability?",
        "doc_id": "test_doc"
    }

@pytest.fixture
def analyze_request_data():
    """Prepare analyze request data"""
    return {
        "text": "We commit to reducing emissions by 50% by 2030 and increasing renewable energy usage."
    }

@pytest.fixture
def esg_request_data():
    """Prepare ESG request data"""
    return {
        "text": "Our company is committed to reducing carbon emissions and improving environmental sustainability."
    }

@pytest.fixture
def consistency_request_data():
    """Fixture for consistency request data"""
    return {
        "text": "Our sustainability goals are clear and consistent with our long-term strategy.",
        "doc_id": None
    }

@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.mark.asyncio
async def test_upload_endpoint(client, test_file):
    """Test file upload endpoint"""
    with open(test_file, 'rb') as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "analysis" in data
        assert "esg" in data

@pytest.mark.asyncio
async def test_analyze_endpoint(client, analyze_request_data):
    """Test analyze endpoint"""
    response = client.post("/analyze", json=analyze_request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "analysis" in data
    assert all(key in data["analysis"] for key in [
        "commitment_probability",
        "specificity_probability",
        "cheap_talk_probability",
        "safe_talk_probability"
    ])

@pytest.mark.asyncio
async def test_esg_endpoint(client, esg_request_data):
    """Test ESG endpoint"""
    response = client.post("/esg", json=esg_request_data)
    assert response.status_code == 200
    data = response.json()
    assert all(category in data for category in [
        'Business Ethics & Values', 'Climate Change', 'Community Relations',
        'Corporate Governance', 'Human Capital', 'Natural Capital', 'Non-ESG',
        'Pollution & Waste', 'Product Liability'
    ])

@pytest.mark.asyncio
async def test_consistency_endpoint(client, consistency_request_data):
    """Test consistency endpoint"""
    response = client.post("/consistency/consistency", json=consistency_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "analysis" in data
    analysis = data["analysis"]
    assert all(key in analysis for key in ["consistency_score", "readability_score", "clarity_score"])
    assert all(isinstance(score, float) and 0 <= score <= 1 for score in analysis.values())

@pytest.mark.asyncio
async def test_chat_endpoint_with_document(client, chat_request_data):
    """Test chat endpoint with document"""
    # First upload a document
    with open("tests/test_document.txt", 'rb') as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        upload_response = client.post("/upload", files=files)
        assert upload_response.status_code == 200
        doc_id = upload_response.json()["filename"]
    
    # Then test chat with the document
    chat_request_data["doc_id"] = doc_id
    response = client.post("/api/chat", json=chat_request_data)
    assert response.status_code == 200
    assert "response" in response.json()

@pytest.mark.asyncio
async def test_analyze_endpoint_no_text(client):
    """Test analyze endpoint with no text"""
    # Test with missing text field
    response = client.post("/analyze", json={})
    assert response.status_code == 422  # Validation error
    error_data = response.json()
    assert "status" in error_data
    assert error_data["status"] == "error"
    assert "code" in error_data
    assert error_data["code"] == "VALIDATION_ERROR"

    # Test with empty text
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "analysis" in data
    assert all(0 <= score <= 1 for score in data["analysis"].values())