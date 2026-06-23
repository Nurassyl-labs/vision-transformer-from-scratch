import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on CIFAR-10")
    parser.add_argument("--use_gap", action="store_true", help="Use Global Average Pooling instead of Class Token")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay weight")
    parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for logging")
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def main():
    args = get_args()
    
    # Create directories for saving logs and checkpoints
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Select compute device (MPS GPU for Mac, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # CIFAR-10 data loaders with training data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Initialize ViT-Lite model
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.1,
        use_gap=args.use_gap
    )
    model = model.to(device)
    
    print(f"Model Configuration: use_gap={model.use_gap}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, testloader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)
        
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(), 
                f"checkpoints/best_{args.exp_name}.pth"
            )
            
    # Save training logs to JSON
    log_file_path = f"logs/{args.exp_name}_history.json"
    with open(log_file_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training completed. Logs saved to {log_file_path}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
