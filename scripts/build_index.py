import sys
import os
from pathlib import Path
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

# Add app directory to sys.path to import feature_extractor
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.feature_extractor import FeatureExtractor

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parents[1] / "embeddings"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]

def main():
    """
    Extracts features from all images in the data directory and builds a FAISS index.
    """
    # --- Setup ---
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    feature_extractor = FeatureExtractor()
    
    image_paths = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not image_paths:
        print(f"No images found in {DATA_DIR}. Please add images to the data directory.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- Feature Extraction ---
    all_features = []
    valid_image_paths = []
    
    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            img = Image.open(img_path)
            features = feature_extractor.extract(img)
            all_features.append(features)
            valid_image_paths.append(img_path.name)
        except Exception as e:
            print(f"Could not process {img_path}: {e}")

    if not all_features:
        print("No features were extracted. Exiting.")
        return

    features_np = np.vstack(all_features)
    
    # --- Build FAISS Index ---
    print("Building FAISS index...")
    dimension = features_np.shape[1]
    # Using IndexFlatL2, which is exact search. For very large datasets,
    # consider an approximate index like 'IndexIVFFlat'.
    index = faiss.IndexFlatL2(dimension)
    index.add(features_np)
    
    # --- Save Artifacts ---
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    index_path = EMBEDDINGS_DIR / "faiss.index"
    image_list_path = EMBEDDINGS_DIR / "image_list.txt"

    np.save(embeddings_path, features_np)
    faiss.write_index(index, str(index_path))
    
    with open(image_list_path, "w") as f:
        f.write("\n".join(valid_image_paths))
        
    print(f"Successfully saved {len(valid_image_paths)} embeddings and index.")
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"FAISS index saved to: {index_path}")
    print(f"Image list saved to: {image_list_path}")

if __name__ == "__main__":
    main()
