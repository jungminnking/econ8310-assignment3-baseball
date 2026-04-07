import torch
from assignment_script import BaseballCNN

#saved weights
WEIGHTS = r"\\JUNGMINN\Users\jungm\Documents\GitHub\econ8310-assignment3-baseball\saved_weights.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = BaseballCNN().to(device)

dummy = torch.zeros(1, 3, 8, 64, 64).to(device)
model(dummy)

model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
print("Weights loaded successfully")

with torch.no_grad():
    sample = torch.zeros(1, 3, 8, 64, 64).to(device)
    out    = model(sample)
    pred   = out.argmax(1).item()
    print(f"Predicted class: {'Moving (Pitched)' if pred == 1 else 'Stationary'}")
