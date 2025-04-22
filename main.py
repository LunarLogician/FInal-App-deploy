# main.py

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from analyze.consistency import router as consistency_router
from typing import Optional, Dict, Any
import os
import shutil
from pydantic import BaseModel, ValidationError
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import langid
import math
from openai import OpenAI
import traceback
import textstat
import numpy as np
import uuid

# Initialize document cache
document_cache = {}

app = FastAPI(title="Combined API Service")

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "code": "HTTP_ERROR",
            "status": "error"
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "message": str(exc),
            "code": "VALIDATION_ERROR",
            "status": "error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "message": str(exc),
            "code": "INTERNAL_ERROR",
            "status": "error"
        }
    )

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "https://final-app-deploy-6d92.onrender.com",  # For deployed backend
        "https://final-app-deploy-2.onrender.com"  # For deployed frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Model definitions
class ConsistencyInput(BaseModel):
    chunks: list[str]

# Create router for consistency endpoints
consistency_router = APIRouter()

@consistency_router.post("/")
async def analyze_consistency(input: ConsistencyInput):
    try:
        if not input.chunks or not any(chunk.strip() for chunk in input.chunks):
            raise HTTPException(
                status_code=400,
                detail="No text provided for analysis"
            )
        
        # Join chunks for analysis
        text = " ".join(input.chunks)
        
        # Calculate readability score
        readability_score = calculate_readability(text)
        
        # Calculate clarity score
        clarity_score = calculate_clarity(text)
        
        # Calculate consistency score as average of readability and clarity
        consistency_score = (readability_score + clarity_score) / 2.0
        
        return {
            "status": "success",
            "consistency_score": float(consistency_score)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Consistency Analysis Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the consistency router
app.include_router(consistency_router, prefix="/consistency", tags=["consistency"])

# Load models at startup
print("Loading models...")
# ClimateBERT models for commitment and specificity
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# Global variables for lazy loading
commitment_model = None
specificity_model = None
esg_model = None
commitment_tokenizer = None
specificity_tokenizer = None
esg_tokenizer = None

def load_models():
    """Lazy load models only when needed"""
    global commitment_model, specificity_model, esg_model
    global commitment_tokenizer, specificity_tokenizer, esg_tokenizer
    
    try:
        if commitment_model is None:
            print("Loading commitment model...")
            commitment_model = AutoModelForSequenceClassification.from_pretrained(
                "climatebert/distilroberta-base-climate-commitment"
            ).to(device)
            commitment_tokenizer = AutoTokenizer.from_pretrained(
                "climatebert/distilroberta-base-climate-commitment"
            )
            
        if specificity_model is None:
            print("Loading specificity model...")
            specificity_model = AutoModelForSequenceClassification.from_pretrained(
                "climatebert/distilroberta-base-climate-specificity"
            ).to(device)
            specificity_tokenizer = AutoTokenizer.from_pretrained(
                "climatebert/distilroberta-base-climate-specificity"
            )
            
        if esg_model is None:
            print("Loading ESG model...")
            model_name = "yiyanghkust/finbert-esg-9-categories"
            esg_tokenizer = AutoTokenizer.from_pretrained(model_name)
            esg_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load models: {str(e)}"
        )

def get_score(model, tokenizer, text: str) -> float:
    try:
        # Input validation
        if not isinstance(text, str):
            print(f"Invalid text type: {type(text)}")
            raise ValueError("Invalid text type")
        
        if not text.strip():
            print("Empty text provided")
            raise ValueError("Empty text provided")
        
        # Move model to device
        model.to(device)
        
        # Tokenize with truncation
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            score = float(probs[0, 1].item())  # Get probability of positive class
            
            # Ensure score is a valid float between 0 and 1
            if not isinstance(score, float) or math.isnan(score):
                print(f"Invalid score generated: {score}")
                raise ValueError("Invalid score generated")
                
            print(f"Generated score: {score}")
            return max(0.0, min(1.0, score))
            
    except Exception as e:
        print(f"Error in get_score: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to get score: {str(e)}")

class ESGInput(BaseModel):
    text: str

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    original_name: str
    size: int
    text: str
    analysis: dict
    esg: dict

# ESG categories
ESG_CATEGORIES = [
    'Business Ethics & Values', 'Climate Change', 'Community Relations',
    'Corporate Governance', 'Human Capital', 'Natural Capital', 'Non-ESG',
    'Pollution & Waste', 'Product Liability'
]

@app.post("/esg")
async def analyze_esg(input: ESGInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
            
        # Initialize classifier with top_k=None to get scores for all categories
        classifier = pipeline("text-classification", model=esg_model, tokenizer=esg_tokenizer, device=device, top_k=None)
        
        # Process text with truncation
        results = classifier(input.text, truncation=True, max_length=512)
        
        # Convert to category-score pairs and ensure they are valid floats
        scores = {}
        for pred in results[0]:
            score = float(pred['score'])
            if not (isinstance(score, float) and score >= 0 and score <= 1):
                score = 0.0
            scores[pred['label']] = score
            
        # Ensure all categories have a score
        for category in ESG_CATEGORIES:
            if category not in scores:
                scores[category] = 0.0
                
        # Normalize scores to ensure they sum to 1
        total = sum(scores.values())
        if total > 0:  # Prevent division by zero
            scores = {k: float(v/total) for k, v in scores.items()}
        
        # Format scores as percentages
        formatted_scores = {}
        for category in ESG_CATEGORIES:
            score = scores.get(category, 0.0)
            formatted_scores[category] = max(0.0, min(1.0, float(score)))
            
        print(f"ESG Analysis completed successfully: {formatted_scores}")
        return formatted_scores
        
    except Exception as e:
        print(f"ESG Analysis Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Return default scores instead of error
        return {category: 0.0 for category in ESG_CATEGORIES}

class PineconeInput(BaseModel):
    text: str
    namespace: str

@app.post("/pinecone/upload")
async def upload_to_pinecone(input: PineconeInput):
    try:
        return {
            "status": "success",
            "message": "Text processed successfully",
            "namespace": input.namespace,
            "text_length": len(input.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main analysis endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure uploads directory exists with proper permissions
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        os.chmod(upload_dir, 0o777)

        # Validate file
        if not file:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No file provided",
                    "code": "NO_FILE"
                }
            )

        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        chunk_size = 1024
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > max_size:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "File size exceeds 10MB limit",
                        "code": "FILE_TOO_LARGE"
                    }
                )
        await file.seek(0)

        # Validate file type
        allowed_extensions = {'.txt', '.pdf', '.doc', '.docx'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}",
                    "code": "INVALID_FILE_TYPE"
                }
            )

        # Generate unique filename
        unique_filename = f"{os.urandom(8).hex()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from the file
        text = ""
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_extension == '.pdf':
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            elif file_extension in ['.doc', '.docx']:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

            # Store in cache
            document_cache[unique_filename] = text
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Failed to extract text from file: {str(e)}",
                    "code": "TEXT_EXTRACTION_FAILED"
                }
            )

        # Perform analysis immediately after text extraction
        try:
            # Get commitment and specificity scores
            commitment_score = get_score(commitment_model, commitment_tokenizer, text)
            specificity_score = get_score(specificity_model, specificity_tokenizer, text)
            
            # Calculate derived scores
            cheap_talk_score = commitment_score * (1 - specificity_score)
            safe_talk_score = (1 - commitment_score) * specificity_score

            # Format analysis results
            analysis_result = {
                "commitment_probability": commitment_score,
                "specificity_probability": specificity_score,
                "cheap_talk_probability": cheap_talk_score,
                "safe_talk_probability": safe_talk_score
            }

            # Get ESG scores
            classifier = pipeline("text-classification", model=esg_model, tokenizer=esg_tokenizer, device=device, top_k=None)
            esg_results = classifier(text, truncation=True, max_length=512)
            
            # Process ESG scores
            esg_scores = {}
            for pred in esg_results[0]:
                score = float(pred['score'])
                if not (isinstance(score, float) and score >= 0 and score <= 1):
                    score = 0.0
                esg_scores[pred['label']] = score
                
            # Ensure all ESG categories have a score
            for category in ESG_CATEGORIES:
                if category not in esg_scores:
                    esg_scores[category] = 0.0
                    
            # Normalize ESG scores
            total = sum(esg_scores.values())
            if total > 0:
                esg_scores = {k: float(v/total) for k, v in esg_scores.items()}

        except Exception as e:
            print(f"Analysis Error during upload: {str(e)}")
            analysis_result = {
                "commitment_probability": 0.0,
                "specificity_probability": 0.0,
                "cheap_talk_probability": 0.0,
                "safe_talk_probability": 0.0
            }
            esg_scores = {category: 0.0 for category in ESG_CATEGORIES}

        return {
            "status": "success",
            "message": "File uploaded and analyzed successfully",
            "filename": unique_filename,
            "original_name": file.filename,
            "size": file_size,
            "text": text,
            "analysis": analysis_result,
            "esg": esg_scores
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "code": "INTERNAL_ERROR"
            }
        )

class AnalyzeInput(BaseModel):
    text: str
    doc_id: Optional[str] = None

# Cheap Talk Analysis patterns
COMMITMENT_PATTERNS = [
    r'will|shall|must|commit|pledge|promise|ensure|guarantee|dedicated to|aim to|target|goal|by \d{4}',
    r'we are committed|we will|we shall|we must|we promise|we guarantee|we ensure'
]

SPECIFICITY_PATTERNS = [
    r'\d+%|\d+ percent|\d+ tonnes|\d+MW|\d+ megawatts|\d+ GW|\d+ gigawatts',
    r'specific|measurable|timebound|quantifiable|detailed|precise|exact|defined',
    r'by \d{4}|by 20\d{2}|in \d{4}|in 20\d{2}'
]

def analyze_cheap_talk(text: str) -> dict:
    import re
    
    # Count commitment indicators
    commitment_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                         for pattern in COMMITMENT_PATTERNS)
    
    # Count specificity indicators
    specificity_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in SPECIFICITY_PATTERNS)
    
    # Normalize scores using word count
    word_count = len(text.split())
    commitment_prob = min(commitment_count / (word_count * 0.05), 1.0)
    specificity_prob = min(specificity_count / (word_count * 0.05), 1.0)
    
    # Calculate cheap talk and safe talk scores
    cheap_talk_prob = commitment_prob * (1 - specificity_prob)
    safe_talk_prob = (1 - commitment_prob) * specificity_prob
    
    return {
        "commitment_probability": commitment_prob,
        "specificity_probability": specificity_prob,
        "cheap_talk_probability": cheap_talk_prob,
        "safe_talk_probability": safe_talk_prob
    }

@app.post("/analyze")
async def analyze_text(input: AnalyzeInput):
    try:
        # Get text from input or document cache
        text = input.text if input.text else ""
        if not text and input.doc_id:
            text = document_cache.get(input.doc_id, '')
        
        # Validate text is not empty
        if not text or not text.strip():
            print("No text provided for analysis")
            return {
                "status": "error",
                "message": "No text provided for analysis"
            }
            
        # Move models to device
        commitment_model.to(device)
        specificity_model.to(device)
        
        try:
            # Get commitment and specificity scores
            commitment_score = get_score(commitment_model, commitment_tokenizer, text)
            specificity_score = get_score(specificity_model, specificity_tokenizer, text)
            
            # Calculate cheap talk and safe talk scores
            cheap_talk_score = commitment_score * (1 - specificity_score)
            safe_talk_score = (1 - commitment_score) * specificity_score
            
            # Return response in the format expected by frontend
            return {
                "status": "success",
                "analysis": {
                    "commitment_probability": float(commitment_score),
                    "specificity_probability": float(specificity_score),
                    "cheap_talk_probability": float(cheap_talk_score),
                    "safe_talk_probability": float(safe_talk_score)
                }
            }
        except Exception as model_error:
            print(f"Model Error: {str(model_error)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "Error processing text with models"
            }
            
    except Exception as e:
        print(f"Analysis Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": "Failed to analyze text"
        }

@app.get("/")
async def root():
    return {"message": "API is running"}

# Add chat input model
class ChatInput(BaseModel):
    message: str
    doc_id: Optional[str] = None

@app.post("/api/chat")
async def chat(input: ChatInput):
    try:
        # Get document context if doc_id is provided
        context = ""
        if input.doc_id:
            doc_content = document_cache.get(input.doc_id, '')
            if doc_content:
                context = f"Document Content: {doc_content}\n\n"
        
        # Get AI response
        response = get_ai_response(context, input.message)
        
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def get_ai_response(context: str, question: str) -> str:
    """Get AI response using OpenAI API"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Construct the prompt
        prompt = f"""Context: {context}
        
Question: {question}

Please provide a detailed answer based on the context above."""
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed answers based on the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in get_ai_response: {str(e)}")
        raise

def calculate_readability(text: str) -> float:
    """Calculate readability score using Flesch Reading Ease"""
    try:
        if not text:
            raise ValueError("Empty text provided")
            
        # Calculate Flesch Reading Ease score
        raw_score = textstat.flesch_reading_ease(text)
        print(f"Raw Flesch score: {raw_score}")
        
        # Normalize score to 0-1 range (Flesch scores typically range from 0-100)
        # Higher scores mean easier to read
        normalized_score = max(0.0, min(1.0, raw_score / 100.0))
        print(f"Normalized readability: {normalized_score}")
        
        return normalized_score
    except Exception as e:
        print(f"Error calculating readability: {str(e)}")
        raise

def calculate_clarity(text: str) -> float:
    """Calculate clarity score using text complexity metrics"""
    try:
        if not text:
            raise ValueError("Empty text provided")
            
        # Tokenize text
        words = text.lower().split()
        total_words = len(words)
        unique_words = len(set(words))
        
        if total_words == 0:
            raise ValueError("No words found in text")
        
        # Calculate lexical diversity (Type-Token Ratio)
        ttr = unique_words / total_words
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / total_words
        
        # Calculate sentence complexity
        sentences = text.split('.')
        avg_sentence_length = total_words / max(1, len(sentences))
        
        # Combine metrics into clarity score
        # Normalize each component
        ttr_norm = min(1.0, ttr * 2)  # TTR typically ranges from 0-0.5
        word_length_norm = max(0.0, min(1.0, 1 - (avg_word_length - 4) / 4))  # Penalize average word length > 8
        sentence_length_norm = max(0.0, min(1.0, 1 - (avg_sentence_length - 10) / 20))  # Penalize very long sentences
        
        # Weighted average of components
        clarity_score = (ttr_norm * 0.4 + word_length_norm * 0.3 + sentence_length_norm * 0.3)
        
        print(f"Clarity metrics:")
        print(f"TTR: {ttr_norm}")
        print(f"Word length: {word_length_norm}")
        print(f"Sentence length: {sentence_length_norm}")
        print(f"Final clarity score: {clarity_score}")
        
        return clarity_score
        
    except Exception as e:
        print(f"Error calculating clarity: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
