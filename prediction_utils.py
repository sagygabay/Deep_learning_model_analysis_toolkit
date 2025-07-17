# prediction_utils.py

import os
import torch
import timm.data
from PIL import Image

def get_prediction_details(model, image_path):
    """
    The single, definitive function to calculate prediction details for an image.
    
    Returns:
        tuple: (predicted_class, confidence, raw_logit) or (None, None, None) on failure.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"    - Warning: Could not open image {os.path.basename(image_path)}. Skipping. Error: {e}")
        return None, None, None
        
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        raw_output = model(input_tensor)

    raw_logit = raw_output.item()
    confidence = torch.sigmoid(raw_output).item()
    predicted_class = 1 if confidence > 0.5 else 0

    return predicted_class, confidence, raw_logit