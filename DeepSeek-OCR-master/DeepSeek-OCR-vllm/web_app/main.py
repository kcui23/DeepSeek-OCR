import asyncio
import os
import sys
import shutil
import uuid
from pathlib import Path
from typing import Optional
import json

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_engine import OCRInferenceEngine

app = FastAPI(title="DeepSeek-OCR Web Interface")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global inference engine
inference_engine: Optional[OCRInferenceEngine] = None


class InferenceRequest(BaseModel):
    image_path: str
    prompt: str
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    min_crops: int = 2
    max_crops: int = 6


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    print("Initializing OCR Inference Engine...")
    inference_engine = OCRInferenceEngine()
    await inference_engine.initialize()
    print("OCR Inference Engine initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global inference_engine
    if inference_engine:
        await inference_engine.cleanup()


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse({
            "success": True,
            "filename": unique_filename,
            "path": str(file_path)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference")
async def run_inference(
    image_path: str = Form(...),
    prompt: str = Form(...),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    crop_mode: bool = Form(True),
    min_crops: int = Form(2),
    max_crops: int = Form(6)
):
    """Run OCR inference on the uploaded image"""
    try:
        if not inference_engine:
            raise HTTPException(status_code=500, detail="Inference engine not initialized")

        # Verify image exists
        img_path = Path(image_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        # Create output directory for this inference
        output_id = uuid.uuid4().hex
        output_path = OUTPUT_DIR / output_id
        output_path.mkdir(exist_ok=True)
        (output_path / "images").mkdir(exist_ok=True)

        # Run inference
        result = await inference_engine.infer(
            image_path=str(img_path),
            prompt=prompt,
            output_path=str(output_path),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            min_crops=min_crops,
            max_crops=max_crops
        )

        # Prepare response
        response = {
            "success": True,
            "output_id": output_id,
            "raw_result": result["raw_result"],
            "processed_result": result["processed_result"],
            "has_visualized_image": result["has_visualized_image"]
        }

        return JSONResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/output/{output_id}/{filename:path}")
async def get_output_file(output_id: str, filename: str):
    """Retrieve output files (images, results, etc.)"""
    file_path = OUTPUT_DIR / output_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "inference_engine_ready": inference_engine is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=19198)
