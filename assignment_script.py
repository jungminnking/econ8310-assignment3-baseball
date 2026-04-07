# Importing...
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# Annotated videos data (XML) 
XML_PATH   = r"\\JUNGMINN\Users\jungm\Documents\GitHub\econ8310-assignment3-baseball\IMG_9197_hugo.xml"
VIDEO_PATH = r"\\JUNGMINN\Users\jungm\Documents\GitHub\econ8310-assignment3-baseball\IMG_9197_hugo.mov"

## Data Loader... 
# *Disclaimer* 
# While this data loading part would be the crux of this assignment, 
# frankly, I couldn't finish this up without the help of an AI tool (I used Claude) due to limited caliber...
# https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html: This was helpful, but didn't really solve my problem.
# So, I've just bounced off the AI of the idea to incorporate annotated datasets given their structure  
# This allowed me to actually study more, rather than demonstrating my current capability; also, believe that this is a good starting point to go futher through the semester project.
# That said, I'm willing to accept any deductions incurred from using AI tools. Thank you.    
class BaseballDataset(Dataset):
    def __init__(self, xml_path, video_path, n_frames=8, img_size=64):
        self.n_frames  = n_frames
        self.img_size  = img_size
        self.samples   = []
        root             = ET.parse(xml_path).getroot() #parsing and getting root
        self.frames      = self._load_all_frames(video_path) #all frames into memory
        total_frames     = len(self.frames) 

        for track in root.findall('track'):
            boxes = []
            for box in track.findall('box'):
                if int(box.attrib['outside']) == 1:
                    continue #in case that balls turn invisible
                frame_idx = int(box.attrib['frame'])
                if frame_idx >= total_frames:
                    continue #Skips if beyond the actual number of frames
                moving_attr = box.find("attribute[@name='moving']") #moving and stationary
                is_moving   = moving_attr is not None and moving_attr.text.strip().lower() == 'true'
                boxes.append({
                    'frame'  : frame_idx,
                    'xtl'    : float(box.attrib['xtl']),
                    'ytl'    : float(box.attrib['ytl']),
                    'xbr'    : float(box.attrib['xbr']),
                    'ybr'    : float(box.attrib['ybr']),
                    'moving' : is_moving,
                })
            if len(boxes) < n_frames:
                continue #If a track has fewer visible boxes than our window size, we can't build a sample from it — skip it.
            label = 1 if sum(b['moving'] for b in boxes) > len(boxes) / 2 else 0 #Majority vote — if more than half the boxes in this track are moving=true, the whole track gets label 1 (pitched), otherwise 0 (stationary).
            for start in range(0, len(boxes) - n_frames + 1, n_frames // 2):
                self.samples.append((boxes[start:start + n_frames], label)) 
    
    def _load_all_frames(self, video_path):
        cap, frames = cv2.VideoCapture(video_path), [] #Opens the video file with OpenCV and initializes an empty frames list in one line.
        while True:
            ret, frame = cap.read()
            if not ret:
                break #Reads frames one by one. ret is False when the video ends, which breaks the loop.
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #OpenCV loads frames as BGR by default. 
        cap.release()
        return frames 
    
    def _crop_ball(self, frame, box, padding=10):
        h, w = frame.shape[:2] #Gets the height and width of the frame so we can clamp coordinates.
        x1   = max(0, int(box['xtl']) - padding) #Adds 10px padding around the bounding box. max(0,...) and min(w/h,...).
        y1   = max(0, int(box['ytl']) - padding)
        x2   = min(w, int(box['xbr']) + padding)
        y2   = min(h, int(box['ybr']) + padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) #Cuts out the ball region. If the crop is somehow empty, substitutes a black square.
        return cv2.resize(crop, (self.img_size, self.img_size)) #Resizes every crop to the same 64×64 size so all tensors have identical shape.
        
    def __len__(self):
        return len(self.samples) # Required by PyTorch — tells the DataLoader how many total samples exist.

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        crops = [self._crop_ball(self.frames[b['frame']], b) for b in window] #Required by PyTorch — given an index, fetches the window of 8 boxes and crops the ball out of each corresponding frame.
        video = torch.tensor(np.stack(crops).astype(np.float32) / 255.0).permute(3, 0, 1, 2)
        return video, torch.tensor(label, dtype=torch.long)

## Building Neural Network Model
# LeNet to process the image
# Basic LeNet
class BaseballCNN(nn.Module):
    def __init__(self):
        super(BaseballCNN, self).__init__()
         #Two Conv layers (LeNet Structure)
        self.conv1 = nn.LazyConv3d(6,  kernel_size=3, padding=1) #Two 3D conv layers, instead of 2D, consideirng a time dimension
        self.conv2 = nn.LazyConv3d(16, kernel_size=3)
        #Three fully connected layers
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(2) #Final outputs: 2 classes: stationary vs moving

    def forward(self, x):
        # x shape: (B, 3, T, 64, 64)
        x = F.max_pool3d(F.relu(self.conv1(x)), (1, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## Training & Fitting the model
dataset  = BaseballDataset(XML_PATH, VIDEO_PATH, n_frames=8)
train_n  = max(1, int(0.8 * len(dataset))) # 80% of samples go to training
train_set, val_set = random_split(dataset, [train_n, len(dataset) - train_n]) # 20% to validation
train_dataloader   = DataLoader(train_set, batch_size=4, shuffle=True) #shuffle=True, to randomize sample order for each epoch
test_dataloader    = DataLoader(val_set,   batch_size=4)
print(f"Total samples: {len(dataset)}")
print(f"Moving        : {sum(s[1] for s in dataset.samples)}")
print(f"Stationary    : {sum(1-s[1] for s in dataset.samples)}")

# Uses GPU if available, otherwise CPU. 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = BaseballCNN().to(device)

# Initialise lazy layers with one dummy forward pass
dummy = torch.zeros(1, 3, 8, 64, 64).to(device)
model(dummy)

# Fitting through 20 epoch
opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(20): #epoch 20
    model.train()
    total_loss, correct, total = 0, 0, 0
    for videos, labels in train_dataloader:
        videos, labels = videos.to(device), labels.to(device)
        opt.zero_grad() #PyTorch accumulates gradients by default so reset them manually
        out  = model(videos) #Forward pass — runs the batch through the model to get predictions, then computes how wrong they are.
        loss = crit(out, labels)
        loss.backward() #Backward pass — computes gradients via backpropagation. opt.step() uses those gradients to update the weights
        opt.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item() #Tracks loss and accuracy across the epoch. out.argmax(1) picks the class with the highest score.
        total      += len(labels)
#Testing the model
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad(): #turns off gradient computation during validation
        for videos, labels in test_dataloader:
            videos, labels = videos.to(device), labels.to(device)
            val_correct += (model(videos).argmax(1) == labels).sum().item()
            val_total   += len(labels) #Runs validation batches through the model and counts correct predictions.

    print(f"Epoch {epoch+1:02d}/{20}  "
          f"loss={total_loss/len(train_dataloader):.4f}  "
          f"train_acc={correct/total:.2%}  "
          f"val_acc={val_correct/max(val_total,1):.2%}")

# Save weights
WEIGHTS = r"\\JUNGMINN\Users\jungm\Documents\GitHub\econ8310-assignment3-baseball\saved_weights.pth"
torch.save(model.state_dict(), WEIGHTS)
print(f"\nWeights saved → {WEIGHTS}")