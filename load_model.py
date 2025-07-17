import torch
import timm

def LOAD_MODEL(MODEL_PATH):
    """
    Loads a custom-trained 'efficientnet_b3' for a regression task.
    """
    MODEL_ARCH = 'efficientnet_b3'
    NUM_CLASSES = 1 
    
    # --- 1. Load Model ---
    print("Loading model...")
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=NUM_CLASSES)
    
    # --- 2. Load Checkpoint and Fix Keys ---
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cuda'), weights_only=False)
    
    model_state_dict = checkpoint['model_state_dict']
    classifier_state_dict = checkpoint['classifier_state_dict']
    
    # Re-map the classifier keys from the checkpoint to the model's expected names
    model_state_dict['classifier.weight'] = classifier_state_dict['1.weight']
    model_state_dict['classifier.bias'] = classifier_state_dict['1.bias']
    
    # --- 3. Load Weights and Set to Eval Mode ---
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print("Model is loaded successfully!")
    return model