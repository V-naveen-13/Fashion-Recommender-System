import faiss
import numpy as np
from PIL import Image
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from functools import lru_cache

from .feature_extractor import feature_extractor

# --- Configuration ---
APP_DIR = Path(__file__).parent
EMBEDDINGS_DIR = APP_DIR.parent / "embeddings"
DATA_DIR = APP_DIR.parent / "data"
STATIC_DIR = APP_DIR / "static"

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
IMAGE_LIST_PATH = EMBEDDINGS_DIR / "image_list.txt"

# --- FastAPI App Initialization ---
app = FastAPI(title="Fashion Recommender System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Load Resources ---
@lru_cache(maxsize=1)
def load_resources():
    """Load FAISS index, embeddings, and image list into memory."""
    if not (FAISS_INDEX_PATH.exists() and EMBEDDINGS_PATH.exists() and IMAGE_LIST_PATH.exists()):
        raise RuntimeError("Index files not found. Please run scripts/build_index.py first.")
    
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    with open(IMAGE_LIST_PATH, "r") as f:
        image_list = [line.strip() for line in f.readlines()]
        
    return index, image_list

try:
    index, image_list = load_resources()
    print("Successfully loaded FAISS index and image list.")
except RuntimeError as e:
    print(f"Error: {e}")
    index, image_list = None, None

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Serve images from the data directory."""
    image_path = DATA_DIR / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.post("/recommend/")
async def recommend(file: UploadFile = File(...)):
    """
    Receive an image, extract features, and return top 5 similar items.
    """
    if not index or not image_list:
        raise HTTPException(status_code=503, detail="System not ready. Index not loaded.")

    # 1. Validate and read image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Extract features from the query image
    query_features = feature_extractor.extract(query_image)

    # 3. Search the FAISS index
    k = 5  # Number of recommendations to return
    distances, indices = index.search(query_features, k)

    # 4. Format and return results
    results = [
        {"filename": image_list[i], "distance": float(d)}
        for i, d in zip(indices[0], distances[0])
    ]
    
    return {"recommendations": results}
