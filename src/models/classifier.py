"""
Production-ready flower classifier for inference.

Usage:
    from src.models.classifier import FlowerClassifier
    
    classifier = FlowerClassifier("artifacts/models/vit_2stage_best.pt")
    result = classifier.predict("path/to/flower.jpg")
    
    print(result['predicted_class'])   # 'sunflower'
    print(result['confidence'])        # 0.987
    print(result['top_3_predictions']) # [('sunflower', 0.987), ...]
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from typing import Dict, List, Tuple, Optional


class FlowerClassifier:
    """
    Production flower classifier using trained ViT model.
    
    Attributes:
        model: The trained PyTorch model
        transform: Image preprocessing pipeline
        class_names: List of 102 flower class names
        device: CPU or CUDA device
    """
    
    # Oxford 102 Flowers class names
    CLASS_NAMES = [
        'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
        'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise',
        'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea',
        'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
        'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily',
        'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
        'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke',
        'sweet william', 'carnation', 'garden phlox', 'love in the mist',
        'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
        'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy',
        'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue',
        'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
        'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium',
        'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia',
        'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush',
        'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris',
        'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose',
        'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily',
        'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
        'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily',
        'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea',
        'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
        'trumpet creeper', 'blackberry lily'
    ]
    
    def __init__(
        self, 
        model_path: str,
        model_name: str = 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to saved model weights (.pt file)
            model_name: timm model architecture name
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = timm.create_model(model_name, pretrained=False, num_classes=102)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (must match training)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"FlowerClassifier loaded on {self.device}")
    
    def predict(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Predict flower class from image.
        
        Args:
            image_path: Path to flower image
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary containing:
                - predicted_class: str - Most likely class name
                - confidence: float - Confidence score (0-1)
                - top_k_predictions: List[Tuple[str, float]] - Top-k (class, prob) pairs
                - class_index: int - Predicted class index (0-101)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        
        # Format results
        top_k_predictions = [
            (self.CLASS_NAMES[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        predicted_idx = top_indices[0].item()
        
        return {
            'predicted_class': self.CLASS_NAMES[predicted_idx],
            'confidence': top_probs[0].item(),
            'top_k_predictions': top_k_predictions,
            'class_index': predicted_idx
        }
    
    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[Dict]:
        """
        Predict flower classes for multiple images.
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(path, top_k) for path in image_paths]
    
    def get_class_names(self) -> List[str]:
        """Return list of all 102 class names."""
        return self.CLASS_NAMES.copy()


# Convenience function for quick inference
def predict_flower(
    image_path: str,
    model_path: str = 'artifacts/models/vit_2stage_best.pt'
) -> Dict:
    """
    Quick prediction function.
    
    Args:
        image_path: Path to flower image
        model_path: Path to model weights
    
    Returns:
        Prediction dictionary
    """
    classifier = FlowerClassifier(model_path)
    return classifier.predict(image_path)


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'artifacts/models/vit_2stage_best.pt'
        
        classifier = FlowerClassifier(model_path)
        result = classifier.predict(image_path)
        
        print(f"\nPrediction Results:")
        print(f"  Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Top 3:")
        for cls, prob in result['top_k_predictions']:
            print(f"    - {cls}: {prob:.4f}")
    else:
        print("Usage: python classifier.py <image_path> [model_path]")
