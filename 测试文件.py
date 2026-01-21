# infer_bee_ant_fix.py
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os, sys

# --- 关键：先导入 Tudui 类，这样 torch.load 能找到它 ---
#from ants_model import Tudui

class Tudui(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Tudui, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

IMAGE_PATH = 'images/2.png'
MODEL_PATH = 'best_morden.pth'   # 你之前保存的模型文件名
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform 与训练时 val_transform 保持一致
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

img = Image.open(IMAGE_PATH).convert('RGB')
img_t = transform(img).unsqueeze(0).to(DEVICE)

# 现在直接加载完整模型（因为 Tudui 已可见）
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()

with torch.no_grad():
    outputs = model(img_t)
    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(outputs.argmax(dim=1).cpu().numpy()[0])

classes = ['classical_architecture', 'morden_architecture']
print(classes[pred_idx])
print("Raw outputs:", outputs.cpu().numpy())
print("Probabilities:", probs)
print("Predicted:", classes[pred_idx])
