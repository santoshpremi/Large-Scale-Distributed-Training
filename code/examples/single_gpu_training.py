"""
Single GPU Training Example

This example demonstrates basic training techniques that work on single GPU setups.
Shows mixed precision, memory optimization, and performance monitoring.

Compatible with single GPU setups (T4/P100/V100 GPUs)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np


class SingleGPUModel(nn.Module):
    """Model optimized for single GPU training"""
    
    def __init__(self, input_dim=32*32*3, hidden_dim=512, output_dim=10, num_layers=8):
        super(SingleGPUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input layer
        x = self.input_layer(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


def get_device_info():
    """Get device information for single GPU setup"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU: {device_name}")
        print(f"Memory: {device_memory:.1f} GB")
        
        # Set appropriate peak TFLOPs
        if "T4" in device_name:
            peak_tflops = 65.1
        elif "P100" in device_name:
            peak_tflops = 21.2
        elif "V100" in device_name:
            peak_tflops = 125.0
        else:
            peak_tflops = 65.1
        
        print(f"Peak TFLOPs: {peak_tflops}")
        return torch.device('cuda'), peak_tflops
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu'), 0.1


def get_cifar10_dataloader(batch_size=32, num_workers=2):
    """Get CIFAR-10 dataloader optimized for single GPU"""
    
    # Transforms
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
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, use_mixed_precision=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_epoch(model, test_loader, criterion, device, use_mixed_precision=True):
    """Test the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def benchmark_training_speed(model, train_loader, device, use_mixed_precision=True):
    """Benchmark training speed"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Warmup
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i >= 5:  # Warmup for 5 batches
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i, (data, target) in enumerate(train_loader):
        if i >= 50:  # Benchmark for 50 batches
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 50
    return avg_time


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(test_accs, label='Test Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main training function"""
    
    print("Single GPU Training")
    print("=" * 50)
    
    # Get device
    device, peak_tflops = get_device_info()
    print()
    
    # Configuration
    batch_size = 64 if device.type == 'cuda' else 32
    num_epochs = 10
    learning_rate = 0.001
    use_mixed_precision = device.type == 'cuda'
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print()
    
    # Get data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloader(batch_size)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = SingleGPUModel(
        input_dim=32*32*3,
        hidden_dim=512,
        output_dim=10,
        num_layers=8
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Benchmark training speed
    print("Benchmarking training speed...")
    fp32_time = benchmark_training_speed(model, train_loader, device, use_mixed_precision=False)
    fp16_time = benchmark_training_speed(model, train_loader, device, use_mixed_precision=True)
    
    print(f"  FP32 time per batch: {fp32_time:.4f}s")
    print(f"  FP16 time per batch: {fp16_time:.4f}s")
    print(f"  Speedup: {fp32_time/fp16_time:.2f}x")
    print()
    
    # Training loop
    print("Starting training...")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_mixed_precision
        )
        
        # Test
        test_loss, test_acc = test_epoch(
            model, test_loader, criterion, device, use_mixed_precision
        )
        
        # Store results
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # Final summary
    print("\nTraining Complete!")
    print(f"Best test accuracy: {max(test_accs):.2f}%")
    print(f"Final test accuracy: {test_accs[-1]:.2f}%")


if __name__ == "__main__":
    main()
