import warnings
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from datasets import FrameImageDataset, FrameVideoDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)
# allow gpus to go fast
torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.bfloat16)
# initing random weights, need to seed before
from per_frame_model import model

root_dir = '/home/thorh/02516/project2/dataset/ucf101' # leakage
#root_dir = '/home/thorh/02516/project2/dataset/ucf101_noleakage'

transform_train = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform_train)
test_framevideo_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform_test, stack_frames=True)
val_framevideo_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform_test, stack_frames=True)
train_dataloader = DataLoader(train_frameimage_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_framevideo_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_framevideo_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = torch.compile(model)

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = F.cross_entropy(output, labels, label_smoothing=0.2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = output.argmax(1)
        train_correct += (labels==predicted).sum().cpu().item()
    
    epoch_loss = train_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {train_correct/len(train_frameimage_dataset):.4f}')
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for frames, label in val_dataloader:
            label = label.to(device)
            b, c, t, h, w = frames.shape
            frames = frames.squeeze(0).permute(1, 0, 2, 3).to(device)
            outputs = model(frames)
            loss = F.cross_entropy(outputs.mean(dim=0, keepdim=True), label)
            val_loss += loss.item()
            avg_output = outputs.mean(dim=0, keepdim=True)
            predicted = avg_output.argmax(1)
            val_correct += (label == predicted).sum().cpu().item()
    val_loss /= len(val_dataloader)
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_correct/len(val_framevideo_dataset):.4f}')

model.eval()
test_correct = 0
with torch.no_grad():
    for frames, label in test_dataloader:
        label = label.to(device)
        b, c, t, h, w = frames.shape
        frames = frames.squeeze(0).permute(1, 0, 2, 3).to(device)
        outputs = model(frames)
        avg_output = outputs.mean(dim=0, keepdim=True)
        predicted = avg_output.argmax(1)
        test_correct += (label == predicted).sum().cpu().item()
print(f'Test Accuracy: {test_correct/len(test_framevideo_dataset):.4f}')