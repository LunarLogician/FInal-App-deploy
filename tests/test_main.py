import pytest
from fastapi.testclient import TestClient
from main import app
import os
from pathlib import Path
from httpx import AsyncClient

# Test data
TEST_TEXT = "This is a test text for analysis."
TEST_ESG_TEXT = "Our company is committed to reducing carbon emissions and promoting sustainable practices."
TEST_CHAT_MESSAGE = "What are the key points in this document?"

@pytest.fixture
def test_file(tmp_path):
    # Create a temporary test file
    test_file_path = tmp_path / "test.txt"
    test_file_path.write_text(TEST_TEXT)
    return test_file_path

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.mark.asyncio
async def test_upload_endpoint(client, test_file):
    with open(test_file, "rb") as f:
        response = await client.post(
            "/upload",
            files={"file": ("test.txt", f, "text/plain")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] is not None
    assert data["text"] == TEST_TEXT
    assert "analysis" in data
    assert "esg" in data

@pytest.mark.asyncio
async def test_analyze_endpoint(client):
    response = await client.post(
        "/analyze",
        json={"text": TEST_TEXT}
    )
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert "commitment_probability" in data["analysis"]
    assert "specificity_probability" in data["analysis"]
    assert "cheap_talk_probability" in data["analysis"]
    assert "safe_talk_probability" in data["analysis"]

@pytest.mark.asyncio
async def test_esg_endpoint(client):
    response = await client.post(
        "/esg",
        json={"text": TEST_ESG_TEXT}
    )
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

@pytest.mark.asyncio
async def test_chat_endpoint(client):
    # First upload a document to get a doc_id
    with open("tests/test_document.txt", "w") as f:
        f.write(TEST_TEXT)
    
    with open("tests/test_document.txt", "rb") as f:
        upload_response = await client.post(
            "/upload",
            files={"file": ("test_document.txt", f, "text/plain")}
        )
    
    doc_id = upload_response.json()["filename"]
    
    # Test chat with document context
    response = await client.post(
        "/api/chat",
        json={"message": TEST_CHAT_MESSAGE, "doc_id": doc_id}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    
    # Clean up
    os.remove("tests/test_document.txt")

@pytest.mark.asyncio
async def test_analyze_endpoint_no_text(client):
    response = await client.post("/analyze")
    assert response.status_code == 400
    assert "No text provided" in response.json()["message"]

@pytest.mark.asyncio
async def test_consistency_endpoint(client):
    response = await client.post(
        "/consistency/consistency",
        json={"text": TEST_TEXT}
    )
    assert response.status_code == 200
    data = response.json()
    assert "consistency_score" in data
    assert "readability_score" in data
    assert "clarity_score" in data
    assert 0 <= data["consistency_score"] <= 1
    assert 0 <= data["readability_score"] <= 1
    assert 0 <= data["clarity_score"] <= 1 