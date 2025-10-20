import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
from google.cloud import storage
import uvicorn

# Global variables for model
model = None
tokenizer = None
text_generation = None

def download_model_from_gcs(bucket_name: str, gcs_prefix: str, local_path: str):
    """
    Download model files from Google Cloud Storage
    
    Args:
        bucket_name: GCS bucket name
        gcs_prefix: Path prefix in GCS (e.g., 'llama_v10/trained_model')
        local_path: Local directory to save the model
    """
    print(f"Downloading model from gs://{bucket_name}/{gcs_prefix}")
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Create local directory
    os.makedirs(local_path, exist_ok=True)
    
    # List all blobs with the prefix
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    
    downloaded_files = 0
    for blob in blobs:
        # Skip directories
        if blob.name.endswith('/'):
            continue
        
        # Get relative path
        relative_path = blob.name[len(gcs_prefix):].lstrip('/')
        local_file_path = os.path.join(local_path, relative_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download file
        print(f"Downloading: {blob.name} -> {local_file_path}")
        blob.download_to_filename(local_file_path)
        downloaded_files += 1
    
    print(f"Downloaded {downloaded_files} files to {local_path}")
    return local_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for model loading"""
    global model, tokenizer, text_generation
    
    # Startup: Download and load the model
    try:
        # GCS Configuration - Read from environment variables
        bucket_name = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
        gcs_model_path = os.getenv("GCS_MODEL_PATH", "llama_v10/trained_model")
        local_model_path = os.getenv("LOCAL_MODEL_PATH", "./downloaded_model")
        
        print(f"GCS Bucket: {bucket_name}")
        print(f"GCS Path: {gcs_model_path}")
        print(f"Local Path: {local_model_path}")
        
        # Download model from GCS if not already present
        if not os.path.exists(local_model_path) or not os.listdir(local_model_path):
            download_model_from_gcs(bucket_name, gcs_model_path, local_model_path)
        else:
            print(f"Model already exists at {local_model_path}, skipping download")
        
        model_path = Path(local_model_path).resolve()
        
        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")
        
        text_generation = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    yield
    
    # Shutdown: Cleanup if needed
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLM Inference API",
    description="API para generación de texto con modelo personalizado desde GCS",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Pregunta o prompt para el modelo")
    max_tokens: Optional[int] = Field(70, description="Máximo de tokens a generar")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperatura para sampling")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")

class AnswerResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Inference API",
        "status": "active",
        "model_source": "Google Cloud Storage",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "batch_generate": "/batch_generate (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": text_generation is not None,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "gcs_bucket": os.getenv("GCS_BUCKET_NAME", "not_set"),
        "gcs_path": os.getenv("GCS_MODEL_PATH", "not_set")
    }

@app.post("/generate", response_model=AnswerResponse)
async def generate_answer(request: QuestionRequest):
    """Generate answer for a given question"""
    if text_generation is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated = text_generation(
            request.question,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )[0]["generated_text"]
        
        return AnswerResponse(
            question=request.question,
            answer=generated,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/batch_generate")
async def batch_generate(questions: list[str], max_tokens: int = 70):
    """Generate answers for multiple questions"""
    if text_generation is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for question in questions:
            generated = text_generation(
                question,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )[0]["generated_text"]
            
            results.append({
                "question": question,
                "answer": generated
            })
        
        return {"results": results, "status": "success"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answers: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )