import pytest
import httpx
import asyncio
from typing import AsyncGenerator
import os
from pathlib import Path

BASE_URL = "http://localhost:5001"  # Updated to match FastAPI port

@pytest.fixture
def test_file(tmp_path):
    # Create a temporary test file
    test_file_path = tmp_path / "test.txt"
    test_file_path.write_text("Sample document text for testing.")
    return test_file_path

@pytest.fixture
def chat_request_data():
    return {
        "message": "Hello, can you analyze this document?",
        "doc_id": None  # Will be set in the test
    }

@pytest.fixture
def analyze_request_data():
    return {
        "text": "Sample sustainability report text discussing environmental impact and social responsibility."
    }

@pytest.fixture
def esg_request_data():
    return {
        "text": "Our company is committed to reducing carbon emissions and promoting sustainable practices."
    }

@pytest.fixture
def consistency_request_data():
    return {
        "text": "This is a sample text for consistency analysis. It should be clear and readable.",
        "doc_id": None  # Optional field
    }

@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        yield client

@pytest.mark.asyncio
async def test_root_endpoint(client: httpx.AsyncClient):
    """Test root endpoint"""
    try:
        response = await client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_upload_endpoint(client: httpx.AsyncClient, test_file):
    """Test file upload endpoint"""
    try:
        with open(test_file, "rb") as f:
            response = await client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] is not None
        assert "analysis" in data
        assert "esg" in data
        return data["filename"]  # Return the doc_id for other tests
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_analyze_endpoint(client: httpx.AsyncClient, analyze_request_data):
    """Test analyze endpoint"""
    try:
        response = await client.post("/analyze", json=analyze_request_data)
        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert "commitment_probability" in data["analysis"]
        assert "specificity_probability" in data["analysis"]
        assert "cheap_talk_probability" in data["analysis"]
        assert "safe_talk_probability" in data["analysis"]
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_esg_endpoint(client: httpx.AsyncClient, esg_request_data):
    """Test ESG endpoint"""
    try:
        response = await client.post("/esg", json=esg_request_data)
        assert response.status_code == 200
        data = response.json()
        assert "Business Ethics & Values" in data
        assert "Climate Change" in data
        assert "Community Relations" in data
        assert "Corporate Governance" in data
        assert "Human Capital" in data
        assert "Natural Capital" in data
        assert "Non-ESG" in data
        assert "Pollution & Waste" in data
        assert "Product Liability" in data
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_consistency_endpoint(client: httpx.AsyncClient, consistency_request_data):
    """Test consistency endpoint"""
    try:
        response = await client.post("/consistency/consistency", json=consistency_request_data)
        assert response.status_code == 200
        data = response.json()
        assert "consistency_score" in data
        assert "readability_score" in data
        assert "clarity_score" in data
        assert 0 <= data["consistency_score"] <= 1
        assert 0 <= data["readability_score"] <= 1
        assert 0 <= data["clarity_score"] <= 1
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_chat_endpoint_with_document(client: httpx.AsyncClient, chat_request_data, test_file):
    """Test chat endpoint with document context"""
    try:
        # First upload a document
        with open(test_file, "rb") as f:
            upload_response = await client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        doc_id = upload_response.json()["filename"]
        
        # Then test chat with the document
        chat_request_data["doc_id"] = doc_id
        response = await client.post("/api/chat", json=chat_request_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}")

@pytest.mark.asyncio
async def test_analyze_endpoint_no_text(client: httpx.AsyncClient):
    """Test analyze endpoint with no text"""
    try:
        response = await client.post("/analyze")
        assert response.status_code == 400
        assert "No text provided" in response.json()["message"]
    except httpx.RequestError as e:
        pytest.skip(f"Server not available: {str(e)}") 