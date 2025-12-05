# pyright: reportMissingImports=false
try:
    import torch
except Exception as e:
    raise ImportError(
        "The 'torch' package (PyTorch) is required but could not be imported. "
        "Install it with 'pip install torch torchvision' or check your Python environment."
    ) from e
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
class FeatureExtractor:
    def __init__(self):
        # Use the recommended weights for ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        # Remove the final classification layer and use the model as a feature extractor
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        # Use the transforms that were used to train the model
        self.preprocess = weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Feature extractor running on: {self.device}")

    def extract(self, img: Image.Image) -> np.ndarray:
        """
        Extracts a feature vector from an image.
        :param img: PIL.Image.Image - The input image.
        :return: np.ndarray - The L2-normalized feature vector.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature_vector = self.model(input_batch)
            # Flatten the feature vector
            flattened_vector = torch.flatten(feature_vector, 1)
            # L2 Normalize the vector
            normalized_vector = flattened_vector / torch.linalg.norm(flattened_vector, ord=2, dim=1, keepdim=True)
        
        return normalized_vector.cpu().numpy()

# Singleton instance that other parts of the app can import and use
feature_extractor = FeatureExtractor()
