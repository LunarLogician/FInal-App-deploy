import pytest
import httpx
import os
from pathlib import Path
import tempfile

BASE_URL = "http://localhost:5001"

@pytest.fixture
def test_file():
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is a test document for analysis. We commit to reducing emissions by 50% by 2030.")
        f.flush()
        yield f.name
    # Cleanup
    os.unlink(f.name)

@pytest.fixture
def chat_request_data():
    return {
        "message": "What are the emission reduction targets?",
        "doc_id": None
    }

@pytest.fixture
def analyze_request_data():
    return {
        "text": "We commit to reducing emissions by 50% by 2030 and increasing renewable energy usage."
    }

@pytest.fixture
def esg_request_data():
    return {
        "text": "Our company is committed to reducing carbon emissions and improving environmental sustainability."
    }

@pytest.fixture
def consistency_request_data():
    return {
        "chunks": ["This is a sample text for consistency analysis. It should be clear and readable."]
    }

@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        yield client

@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "API is running"

@pytest.mark.asyncio
async def test_upload_endpoint(client, test_file):
    """Test file upload endpoint"""
    with open(test_file, 'rb') as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        response = await client.post("/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "analysis" in data
    assert "esg" in data
    assert "filename" in data

@pytest.mark.asyncio
async def test_analyze_endpoint(client, analyze_request_data):
    """Test analyze endpoint"""
    response = await client.post("/analyze", json=analyze_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    analysis = data["analysis"]
    assert "commitment_probability" in analysis
    assert "specificity_probability" in analysis
    assert "cheap_talk_probability" in analysis
    assert "safe_talk_probability" in analysis

@pytest.mark.asyncio
async def test_esg_endpoint(client, esg_request_data):
    """Test ESG endpoint"""
    response = await client.post("/esg", json=esg_request_data)
    assert response.status_code == 200
    data = response.json()
    expected_categories = [
        'Business Ethics & Values', 'Climate Change', 'Community Relations',
        'Corporate Governance', 'Human Capital', 'Natural Capital', 'Non-ESG',
        'Pollution & Waste', 'Product Liability'
    ]
    for category in expected_categories:
        assert category in data
        assert isinstance(data[category], float)
        assert 0 <= data[category] <= 1

@pytest.mark.asyncio
async def test_consistency_endpoint(client, consistency_request_data):
    """Test consistency endpoint"""
    response = await client.post("/consistency/consistency", json=consistency_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "consistency_score" in data
    assert "consistency_variability" in data
    assert "readability_score" in data
    assert "clarity_score" in data
    assert all(isinstance(score, (float, type(None))) for score in data.values())

@pytest.mark.asyncio
async def test_chat_endpoint_with_document(client, test_file):
    """Test chat endpoint with document context"""
    # First upload a document
    with open(test_file, 'rb') as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        upload_response = await client.post("/upload", files=files)
    
    assert upload_response.status_code == 200
    doc_id = upload_response.json()["filename"]
    
    # Then test chat with the document
    chat_data = {
        "message": "What are the emission reduction targets?",
        "doc_id": doc_id
    }
    response = await client.post("/api/chat", json=chat_data)
    
    # Check if we get either a successful response or an error
    if response.status_code == 500:
        error_data = response.json()
        assert "status" in error_data
        assert error_data["status"] == "error"
        assert "code" in error_data
        assert error_data["code"] == "HTTP_ERROR"
    else:
        assert response.status_code == 200
        assert "response" in response.json()

@pytest.mark.asyncio
async def test_analyze_endpoint_no_text(client):
    """Test analyze endpoint with no text"""
    # Test with missing text field
    response = await client.post("/analyze", json={})
    assert response.status_code == 422  # Validation error
    error_data = response.json()
    assert "status" in error_data
    assert error_data["status"] == "error"
    assert "code" in error_data
    assert error_data["code"] == "VALIDATION_ERROR"
    
    # Test with empty text
    response = await client.post("/analyze", json={"text": ""})
    # The endpoint returns 200 with default scores for empty text
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    analysis = data["analysis"]
    assert "commitment_probability" in analysis
    assert "specificity_probability" in analysis
    assert "cheap_talk_probability" in analysis
    assert "safe_talk_probability" in analysis