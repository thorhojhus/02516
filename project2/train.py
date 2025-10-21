import warnings
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from project2.datasets import FrameVideoDataset, VideoFileDataset, FrameImageDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from project2.models.late_fusion import LateFusionMLP, LateFusionPool
from project2.models.early_fusion import EarlyFusionCNN
from project2.models.per_frame import PerFrameModel
from project2.models.resnet_3d_18 import ResNet3D18
from project2.utils import set_seed, set_default_dtype_based_on_arch

warnings.filterwarnings("ignore", category=UserWarning)
# allow ampere gpus to go fast
torch.set_float32_matmul_precision('high')
#set_default_dtype_based_on_arch()
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#ROOT_DIR = '/dtu/datasets1/02516/ufc10'
#ROOT_DIR = '/dtu/datasets1/02516/ucf101_noleakage'
#ROOT_DIR = '/home/thorh/02516/project2/dataset/ucf101'
#ROOT_DIR = '/home/thorh/02516/project2/dataset/ucf101_noleakage'

def init_transforms(do_train_augmentation=False):
    if do_train_augmentation:
        transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform

def init_model(model_name):
    if model_name == 'early_fusion':
        return EarlyFusionCNN().to(device)
    elif model_name == 'late_fusion_mlp':
        return LateFusionMLP(hidden_dim=256).to(device)
    elif model_name == 'late_fusion_pool':
        return LateFusionPool(feature_dim=2048).to(device) # need to match resnet50 output dim
    elif model_name == 'per_frame':
        return PerFrameModel().to(device)
    elif model_name == 'resnet_3d':
        return ResNet3D18().to(device)
    elif model_name == 'cnn2d_frame':
        from project2.models.cnn_2d_frame import Network
        return Network().to(device)
    elif model_name == 'dualstream':
        from project2.models.dual_stream_network import DualStreamResNet
        return DualStreamResNet().to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def eval_model(model, dataloader, dataset_length, per_frame=False, dualstream=False):
    model.eval()
    correct = 0
    correct_top2 = 0
    eval_loss = 0.0
    if per_frame:
        with torch.inference_mode():
            for frames, label in dataloader:
                label = label.to(device)
                b, c, t, h, w = frames.shape
                frames = frames.squeeze(0).permute(1, 0, 2, 3).to(device)
                outputs = model(frames)
                loss = F.cross_entropy(outputs.mean(dim=0, keepdim=True), label)
                eval_loss += loss.item()
                avg_output = outputs.mean(dim=0, keepdim=True)
                predicted = avg_output.argmax(1)
                correct += (label == predicted).sum().cpu().item()
                _, top2_pred = avg_output.topk(2, dim=1)
                correct_top2 += (top2_pred == label.unsqueeze(1)).any(dim=1).sum().cpu().item()

    elif dualstream:
        with torch.inference_mode():
            for frames, flows, label in dataloader:
                frames, flows, label = frames.to(device), flows.to(device), label.to(device)
                output = model(frames, flows)
                loss = F.cross_entropy(output, label)
                eval_loss += loss.item()
                predicted = output.argmax(1)
                correct += (label == predicted).sum().cpu().item()
                _, top2_pred = output.topk(2, dim=1)
                correct_top2 += (top2_pred == label.unsqueeze(1)).any(dim=1).sum().cpu().item()
    else:
        with torch.inference_mode():
            for frames, label in dataloader:
                frames, label = frames.to(device), label.to(device)
                output = model(frames)
                loss = F.cross_entropy(output, label)
                eval_loss += loss.item()
                predicted = output.argmax(1)
                correct += (label == predicted).sum().cpu().item()
                _, top2_pred = output.topk(2, dim=1)
                correct_top2 += (top2_pred == label.unsqueeze(1)).any(dim=1).sum().cpu().item()
    acc1 = correct / dataset_length
    acc2 = correct_top2 / dataset_length
    avg_loss = eval_loss / len(dataloader)
    return acc1, acc2, avg_loss

def train(model, optimizer, NUM_EPOCHS, train_dataloader, val_dataloader, test_dataloader, per_frame=False, dualstream=False):
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        if dualstream:
            for i, (inputs, flows, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs, flows, labels = inputs.to(device), flows.to(device), labels.to(device)
                output = model(inputs, flows)
                loss = F.cross_entropy(output, labels, label_smoothing=0.2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                predicted = output.argmax(1)
                train_correct += (labels==predicted).sum().cpu().item()
        else:
            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss = F.cross_entropy(output, labels, label_smoothing=0.2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                predicted = output.argmax(1)
                train_correct += (labels==predicted).sum().cpu().item()

        epoch_loss = train_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]\nTrain Loss: {epoch_loss:.4f}, Train Acc: {train_correct/len(train_dataset):.4f}')
        val_acc, val_acc2, val_loss = eval_model(model, val_dataloader, len(val_framevideo_dataset), per_frame=per_frame, dualstream=dualstream)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Acc@2: {val_acc2:.4f}')

    test_acc, test_acc2, test_loss = eval_model(model, test_dataloader, len(test_framevideo_dataset), per_frame=per_frame, dualstream=dualstream)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Acc@2: {test_acc2:.4f}')

if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description="Train a video classification model"

    )
    args.add_argument(
        '--model', type=str, default='early_fusion',
        choices=['early_fusion', 'late_fusion_mlp', 'late_fusion_pool', 'per_frame', 'resnet_3d', 'cnn2d_frame', 'dualstream'],
        help='Model architecture to use'
    )
    args.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs'
    )
    args.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for the optimizer'
    )
    args.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size'
    )
    args.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay for the optimizer'
    )
    args.add_argument(
        '--root_dir', type=str, default='/dtu/datasets1/02516/ucf101_noleakage',
        help='Root directory of the dataset'
    )
    args.add_argument(
        '--compile', action='store_true',
        help='Whether to use torch.compile for model optimization'
    )

    args.add_argument(
        '--train_augmentation', action='store_true',
        help='Whether to use data augmentation during training'
    )

    args = args.parse_args()
    NUM_EPOCHS = args.epochs
    ROOT_DIR = args.root_dir

    model = init_model(args.model)
    per_frame = ("frame" in args.model)
    dualstream = (args.model == "dualstream")

    if args.compile:
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    transform_train = init_transforms(do_train_augmentation=args.train_augmentation)
    transform_test = init_transforms(do_train_augmentation=False)

    if per_frame:
        train_dataset = FrameImageDataset(root_dir=ROOT_DIR, split='train', transform=transform_train)
    elif dualstream:
        from project2.datasets import FrameFlowImageDataset
        train_dataset = FrameFlowImageDataset(root_dir=ROOT_DIR, split='train')
    else:
        train_dataset = FrameVideoDataset(root_dir=ROOT_DIR, split='train', transform=transform_train, stack_frames=True)
    
    if dualstream:
        from project2.datasets import FrameFlowImageDataset
        test_framevideo_dataset = FrameFlowImageDataset(root_dir=ROOT_DIR, split='test')
        val_framevideo_dataset = FrameFlowImageDataset(root_dir=ROOT_DIR, split='val')
    else:
        test_framevideo_dataset = FrameVideoDataset(root_dir=ROOT_DIR, split='test', transform=transform_test, stack_frames=True)
        val_framevideo_dataset = FrameVideoDataset(root_dir=ROOT_DIR, split='val', transform=transform_test, stack_frames=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    if per_frame:
        test_dataloader = DataLoader(test_framevideo_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        val_dataloader = DataLoader(val_framevideo_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    else:
        test_dataloader = DataLoader(test_framevideo_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        val_dataloader = DataLoader(val_framevideo_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)


    train(model, optimizer, NUM_EPOCHS, train_dataloader, val_dataloader, test_dataloader, per_frame=per_frame, dualstream=dualstream)