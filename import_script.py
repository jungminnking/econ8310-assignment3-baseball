import torch
import os
import sys

# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Model
from assignment_script import BaseballCNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BaseballCNN().to(device)
dummy = torch.zeros(1, 3, 8, 64, 64).to(device)
model(dummy)

# Weights
WEIGHTS = os.path.join(BASE_DIR, "saved_weights.pth")
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()