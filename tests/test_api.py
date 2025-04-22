import pytest
import httpx
import os
import tempfile
from main import app, get_ai_response, analyze_cheap_talk
from fastapi.testclient import TestClient
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from httpx import AsyncClient

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
def client():
    """Create a test client for the FastAPI application"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def test_file():
    """Create a temporary test file"""
    content = "We commit to reducing emissions by 50% by 2030 and increasing renewable energy usage."
    with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
        f.write(content)
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
    return {
        "text": "This is a sample text for analysis",
        "doc_id": None
    }

@pytest.fixture
def esg_request_data():
    """Prepare ESG request data"""
    return {
        "text": "Our company is committed to reducing carbon emissions and improving environmental sustainability."
    }

@pytest.fixture
def consistency_request_data():
    return {
        "chunks": [
            "This is the first chunk of text.",
            "This is the second chunk of text.",
            "This is the third chunk of text."
        ]
    }

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://localhost:5001") as client:
        yield client

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_upload_endpoint(client, test_file):
    """Test file upload endpoint"""
    with open(test_file, 'rb') as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "message" in data
        assert "analysis" in data
        assert isinstance(data["analysis"], dict)
        assert "esg" in data
        assert isinstance(data["esg"], dict)

@pytest.mark.asyncio
async def test_analyze_endpoint(async_client, analyze_request_data):
    response = await async_client.post("/analyze", json=analyze_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "commitment_score" in data
    assert "specificity_score" in data
    assert "cheap_talk_score" in data
    assert "safe_talk_score" in data
    assert isinstance(data["commitment_score"], float)
    assert isinstance(data["specificity_score"], float)
    assert isinstance(data["cheap_talk_score"], float)
    assert isinstance(data["safe_talk_score"], float)
    assert 0 <= data["commitment_score"] <= 1
    assert 0 <= data["specificity_score"] <= 1
    assert 0 <= data["cheap_talk_score"] <= 1
    assert 0 <= data["safe_talk_score"] <= 1

def test_esg_endpoint(client, esg_request_data):
    """Test ESG endpoint"""
    response = client.post("/esg", json=esg_request_data)
    assert response.status_code == 200
    data = response.json()
    assert all(category in data for category in [
        'Business Ethics & Values', 'Climate Change', 'Community Relations',
        'Corporate Governance', 'Human Capital', 'Natural Capital', 'Non-ESG',
        'Pollution & Waste', 'Product Liability'
    ])
    assert all(isinstance(score, float) and 0 <= score <= 1 for score in data.values())

@pytest.mark.asyncio
async def test_consistency_endpoint(async_client, consistency_request_data):
    response = await async_client.post("/consistency", json=consistency_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "consistency_score" in data
    assert isinstance(data["consistency_score"], float)
    assert 0 <= data["consistency_score"] <= 1

@pytest.mark.asyncio
async def test_consistency_endpoint_no_text(async_client):
    response = await async_client.post("/consistency", json={"chunks": []})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "No text chunks provided for analysis" in data["detail"]

@pytest.mark.asyncio
async def test_analyze_endpoint_no_text(async_client):
    response = await async_client.post("/analyze", json={"text": "", "doc_id": None})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "No text provided for analysis" in data["detail"]

def test_analyze_cheap_talk():
    """Test analyze_cheap_talk function directly"""
    test_text = "We commit to reducing emissions by 50% by 2030 and increasing renewable energy usage."
    result = analyze_cheap_talk(test_text)
    assert all(key in result for key in [
        "commitment_probability",
        "specificity_probability",
        "cheap_talk_probability",
        "safe_talk_probability"
    ])
    assert all(isinstance(score, float) and 0 <= score <= 1 for score in result.values())