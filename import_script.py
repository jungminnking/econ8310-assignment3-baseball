import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "saved_weights.pth")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import model
from assignment_script import BaseballCNN
model = BaseballCNN().to(device)

# Initialize lazy layers 
dummy = torch.zeros(1, 3, 8, 64, 64).to(device)
model(dummy)

# Load saved weights
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

# Run a quick test inference
with torch.no_grad():
    sample = torch.zeros(1, 3, 8, 64, 64).to(device)
    out = model(sample)
    pred = out.argmax(1).item()
    print(f"Predicted class: {'Moving (Pitched)' if pred == 1 else 'Stationary'}")