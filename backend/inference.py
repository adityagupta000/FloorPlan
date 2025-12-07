import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ResNet50_UNet_Attention


# Configuration - MATCH TRAINING EXACTLY
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# CRITICAL FIX 1: Use 512x512 (same as training!)
# ========================================
TARGET_SIZE = 512  # Changed from 256 to 512

# Image preprocessing (EXACT same as training)
transform = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Load model once
model = None


def load_model():
    """Load trained segmentation model from checkpoint"""
    global model
    if model is None:
        print("Loading segmentation model...")
        
        # Create model (same architecture as training)
        model = ResNet50_UNet_Attention(num_classes=NUM_CLASSES).to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Load ONLY the model_state_dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        model.eval()
        print(f"Model loaded successfully! Using {TARGET_SIZE}x{TARGET_SIZE} resolution")
    
    return model


# ========================================
# CRITICAL FIX 2: Add Test-Time Augmentation (TTA)
# This handles rotation invariance for windows!
# ========================================
def predict_with_tta(image_tensor, model, device):
    """
    Test-Time Augmentation: Predict on multiple rotations and average results.
    This is CRITICAL for detecting windows at any angle!
    
    Args:
        image_tensor: Preprocessed image tensor [C, H, W]
        model: Trained segmentation model
        device: torch device
    
    Returns:
        prediction: Segmented mask [H, W] with class labels
    """
    predictions = []
    
    # Test on original + 3 rotations (0°, 90°, 180°, 270°)
    angles = [0, 90, 180, 270]
    
    with torch.no_grad():
        for angle in angles:
            # Rotate image
            img_rotated = transforms.functional.rotate(image_tensor, angle)
            img_batch = img_rotated.unsqueeze(0).to(device)
            
            # Predict
            output = model(img_batch)
            output = F.interpolate(
                output, 
                size=(TARGET_SIZE, TARGET_SIZE), 
                mode='bilinear', 
                align_corners=False
            )
            output_soft = torch.softmax(output, dim=1)
            
            # Rotate prediction BACK to original orientation
            output_soft = output_soft.squeeze(0)  # Remove batch dim [C, H, W]
            
            # Rotate each class channel back
            rotated_back = []
            for c in range(output_soft.shape[0]):
                channel = output_soft[c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                channel_rotated = transforms.functional.rotate(channel, -angle, fill=0)
                rotated_back.append(channel_rotated.squeeze(0).squeeze(0))
            
            output_soft_back = torch.stack(rotated_back, dim=0)  # [C, H, W]
            predictions.append(output_soft_back)
    
    # Average all predictions (ensemble)
    final_pred_soft = torch.stack(predictions).mean(dim=0)  # [C, H, W]
    final_pred = torch.argmax(final_pred_soft, dim=0)  # [H, W]
    
    return final_pred.cpu().numpy()


def run_segmentation(input_image_path, output_mask_path):
    """
    Run segmentation inference on an input image with TTA for better accuracy.
    """
    try:
        model = load_model()
        
        # Load and preprocess image
        image = Image.open(input_image_path).convert("RGB")
        
        # Apply same preprocessing as training
        image_tensor = transform(image)
        
        # ========================================
        # CRITICAL: Use TTA for rotation-invariant predictions
        # ========================================
        print("Running inference with Test-Time Augmentation (4 rotations)...")
        pred_mask = predict_with_tta(image_tensor, model, DEVICE)
        
        # Keep only allowed classes
        allowed = {0, 1, 2, 3, 4}
        pred_mask = np.where(np.isin(pred_mask, list(allowed)), pred_mask, 0)
        
        # Save mask at FULL resolution (512x512 or original size)
        mask_image = Image.fromarray(pred_mask.astype(np.uint8))
        mask_image.save(output_mask_path)
        
        print(f"Segmentation mask saved: {output_mask_path}")
        print(f"   Resolution: {pred_mask.shape}")
        print(f"   Classes detected: {np.unique(pred_mask)}")
        return True
    
    except Exception as e:
        print(f"❌ Segmentation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Load model immediately when module imports
load_model()