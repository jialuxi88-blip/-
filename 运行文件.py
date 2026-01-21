import os
import time
import shutil
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

# ---------------------------
# 固定随机种子（保证可复现）
# ---------------------------
torch.manual_seed(42)

# ---------------------------
# 定义模型（使用预训练 ResNet18）
# ---------------------------
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
# ---------------------------
# 自定义 Dataset
# ---------------------------
class myData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        """
        root_dir: 'dataset/train' 或 'dataset/val'
        label_dir: 'ants_image' 或 'bees_image'
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = sorted(os.listdir(self.path))
        self.transform = transform

        # 标签映射
        self.label_map = {'classical_image': 0, 'morden_image': 1}
        if self.label_dir not in self.label_map:
            raise ValueError(f"Unknown label_dir: {self.label_dir}")
        self.label = self.label_map[self.label_dir]

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.label

    def __len__(self):
        return len(self.img_path)
# ---------------------------
# 数据增强
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

# ---------------------------
# 数据集与 DataLoader
# ---------------------------
train_root = 'dataset/train'
test_root = 'dataset/val'

train_classical_dataset = myData(train_root, 'classical_image', transform=train_transform)
train_morden_dataset = myData(train_root, 'morden_image', transform=train_transform)

test_classical_dataset = myData(test_root, 'classical_image', transform=val_transform)
test_morden_dataset = myData(test_root, 'morden_image', transform=val_transform)

train_dataset = ConcatDataset([train_classical_dataset, train_morden_dataset])
test_dataset = ConcatDataset([test_classical_dataset, test_morden_dataset])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------
# 路径与设备
# ---------------------------
log_dir = 'logs_train'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print(f"已清空旧日志目录: {log_dir}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

print('训练数据集长度：', len(train_dataset))
print('验证数据集长度：', len(test_dataset))
print("calssical count (train):", len(train_classical_dataset))
print("morden count (train):", len(train_morden_dataset))
print("classical count (val):", len(test_classical_dataset))
print("morden count (val):", len(test_morden_dataset))

# 检查一个 batch 的标签
for imgs, targets in train_loader:
    print("Batch 标签示例:", targets[:16])
    break

# ---------------------------
# 模型、损失函数、优化器
# ---------------------------
model = Tudui(num_classes=2, pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ---------------------------
# 训练超参数
# ---------------------------
epochs = 30
writer = SummaryWriter(log_dir)
start_time = time.time()

best_val_acc = 0.0
patience = 6
no_improve = 0

# ---------------------------
# 训练循环
# ---------------------------
for epoch in range(1, epochs + 1):
    print(f"\n------- Epoch {epoch}/{epochs} -------")
    model.train()
    train_running_loss = 0.0
    train_running_corrects = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        train_running_corrects += (preds == targets).sum().item()

    epoch_loss = train_running_loss / len(train_dataset)
    epoch_acc = train_running_corrects / len(train_dataset)
    writer.add_scalar('train_loss', epoch_loss, epoch)
    writer.add_scalar('train_accuracy', epoch_acc, epoch)
    print(f"Train loss: {epoch_loss:.4f}  Train acc: {epoch_acc:.4f}")

    # ---------------------------
    # 验证阶段
    # ---------------------------
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            val_running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_running_corrects += (preds == targets).sum().item()

    val_loss = val_running_loss / len(test_dataset)
    val_acc = val_running_corrects / len(test_dataset)
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_accuracy', val_acc, epoch)
    print(f"Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")

    scheduler.step(val_acc)

    # ---------------------------
    # 保存最佳模型
    # ---------------------------
    if val_acc > best_val_acc + 1e-6:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model, 'best_morden.pth')
        print(f"✅ Best model saved (val_acc={best_val_acc:.4f})")
    else:
        no_improve += 1
        print(f"No improvement count: {no_improve}/{patience}")

    if no_improve >= patience:
        print("🛑 Early stopping triggered.")
        break

end_time = time.time()
print(f"Training finished. Time elapsed: {end_time - start_time:.1f}s. Best val acc: {best_val_acc:.4f}")
writer.close()