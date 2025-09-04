"""
Model Flops Utilization (MFU) Benchmarking

This example demonstrates how to benchmark and optimize Model Flops Utilization
for distributed training systems.
"""

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


class BenchmarkModel(nn.Module):
    """Model for MFU benchmarking"""
    
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=10, num_layers=10):
        super(BenchmarkModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
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
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x


def calculate_model_flops(model, input_data):
    """Calculate theoretical FLOPS for the model"""
    total_flops = 0
    
    # Input layer
    input_flops = input_data.numel() * model.input_layer.in_features * model.input_layer.out_features
    total_flops += input_flops
    
    # Hidden layers
    for layer in model.hidden_layers:
        linear_layer = layer[0]  # Get the Linear layer
        layer_flops = input_data.numel() * linear_layer.in_features * linear_layer.out_features
        total_flops += layer_flops
    
    # Output layer
    output_flops = input_data.numel() * model.output_layer.in_features * model.output_layer.out_features
    total_flops += output_flops
    
    # Backward pass (approximately 2x forward pass)
    total_flops *= 3  # Forward + Backward + Gradient computation
    
    return total_flops


def measure_training_time(model, input_data, target, criterion, optimizer, num_iterations=100):
    """Measure actual training time"""
    
    # Warmup
    for _ in range(10):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Time measurement
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def calculate_mfu(model, input_data, target, criterion, optimizer, device_peak_tflops):
    """Calculate Model Flops Utilization"""
    
    # Calculate theoretical FLOPS
    theoretical_flops = calculate_model_flops(model, input_data)
    
    # Calculate theoretical time
    theoretical_time = theoretical_flops / (device_peak_tflops * 1e12)
    
    # Measure actual time
    actual_time = measure_training_time(model, input_data, target, criterion, optimizer)
    
    # Calculate MFU
    mfu = theoretical_time / actual_time
    
    return mfu, theoretical_flops, theoretical_time, actual_time


def benchmark_mfu_vs_batch_size(model, device, device_peak_tflops, max_batch_size=1024):
    """Benchmark MFU vs batch size"""
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    mfu_results = []
    
    for batch_size in batch_sizes:
        if batch_size > max_batch_size:
            break
            
        try:
            # Create test data
            input_data = torch.randn(batch_size, model.input_layer.in_features, device=device)
            target = torch.randint(0, model.output_layer.out_features, (batch_size,), device=device)
            
            # Create optimizer and criterion
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Calculate MFU
            mfu, theoretical_flops, theoretical_time, actual_time = calculate_mfu(
                model, input_data, target, criterion, optimizer, device_peak_tflops
            )
            
            mfu_results.append({
                'batch_size': batch_size,
                'mfu': mfu,
                'theoretical_flops': theoretical_flops,
                'theoretical_time': theoretical_time,
                'actual_time': actual_time
            })
            
            print(f"Batch size: {batch_size:4d}, MFU: {mfu:.2%}, "
                  f"Time: {actual_time:.4f}s, FLOPS: {theoretical_flops/1e9:.2f}G")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: Out of memory")
                break
            else:
                raise e
    
    return mfu_results


def benchmark_mfu_vs_model_size(device, device_peak_tflops, batch_size=32):
    """Benchmark MFU vs model size"""
    
    hidden_dims = [512, 1024, 2048, 4096, 8192]
    num_layers_list = [5, 10, 20, 40]
    mfu_results = []
    
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            try:
                # Create model
                model = BenchmarkModel(
                    input_dim=1024,
                    hidden_dim=hidden_dim,
                    output_dim=10,
                    num_layers=num_layers
                ).to(device)
                
                # Create test data
                input_data = torch.randn(batch_size, 1024, device=device)
                target = torch.randint(0, 10, (batch_size,), device=device)
                
                # Create optimizer and criterion
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Calculate MFU
                mfu, theoretical_flops, theoretical_time, actual_time = calculate_mfu(
                    model, input_data, target, criterion, optimizer, device_peak_tflops
                )
                
                mfu_results.append({
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'mfu': mfu,
                    'theoretical_flops': theoretical_flops,
                    'theoretical_time': theoretical_time,
                    'actual_time': actual_time
                })
                
                print(f"Hidden dim: {hidden_dim:4d}, Layers: {num_layers:2d}, "
                      f"MFU: {mfu:.2%}, Time: {actual_time:.4f}s")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Hidden dim {hidden_dim}, Layers {num_layers}: Out of memory")
                    break
                else:
                    raise e
    
    return mfu_results


def benchmark_mixed_precision(model, input_data, target, device_peak_tflops):
    """Benchmark MFU with mixed precision"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup
    for _ in range(10):
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Time measurement
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate MFU
    theoretical_flops = calculate_model_flops(model, input_data)
    theoretical_time = theoretical_flops / (device_peak_tflops * 1e12)
    actual_time = (end_time - start_time) / 100
    
    mfu = theoretical_time / actual_time
    
    return mfu, theoretical_flops, theoretical_time, actual_time


def plot_mfu_results(mfu_results, title="MFU Benchmark Results"):
    """Plot MFU benchmark results"""
    
    if not mfu_results:
        print("No results to plot")
        return
    
    # Extract data
    if 'batch_size' in mfu_results[0]:
        # Batch size benchmark
        batch_sizes = [r['batch_size'] for r in mfu_results]
        mfus = [r['mfu'] for r in mfu_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, mfus, 'bo-')
        plt.xlabel('Batch Size')
        plt.ylabel('Model Flops Utilization (MFU)')
        plt.title(title)
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        
    elif 'hidden_dim' in mfu_results[0]:
        # Model size benchmark
        hidden_dims = [r['hidden_dim'] for r in mfu_results]
        mfus = [r['mfu'] for r in mfu_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(hidden_dims, mfus, 'ro-')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Model Flops Utilization (MFU)')
        plt.title(title)
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
    
    plt.tight_layout()
    plt.show()


def get_device_specs():
    """Get device specifications for single GPU setup"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Set peak TFLOPs based on common single GPU setups
        if "T4" in device_name:
            peak_tflops = 65.1  # T4 tensor cores
        elif "P100" in device_name:
            peak_tflops = 21.2  # P100
        elif "V100" in device_name:
            peak_tflops = 125.0  # V100 tensor cores
        else:
            peak_tflops = 65.1  # Default to T4
        
        print(f"GPU: {device_name}")
        print(f"Memory: {device_memory:.1f} GB")
        print(f"Peak TFLOPs: {peak_tflops}")
        
        return torch.device('cuda'), peak_tflops
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu'), 0.1  # Very low for CPU


def main():
    """Main benchmarking function"""
    
    # Configuration
    device, device_peak_tflops = get_device_specs()
    
    print(f"Running MFU benchmarks on {device}")
    print(f"Device peak TFLOPs: {device_peak_tflops}")
    
    # Create model
    model = BenchmarkModel(
        input_dim=1024,
        hidden_dim=2048,
        output_dim=10,
        num_layers=10
    ).to(device)
    
    # Benchmark 1: MFU vs Batch Size
    print("\n=== Benchmark 1: MFU vs Batch Size ===")
    batch_size_results = benchmark_mfu_vs_batch_size(model, device, device_peak_tflops)
    
    # Benchmark 2: MFU vs Model Size
    print("\n=== Benchmark 2: MFU vs Model Size ===")
    model_size_results = benchmark_mfu_vs_model_size(device, device_peak_tflops)
    
    # Benchmark 3: Mixed Precision
    print("\n=== Benchmark 3: Mixed Precision ===")
    input_data = torch.randn(32, 1024, device=device)
    target = torch.randint(0, 10, (32,), device=device)
    
    # FP32
    mfu_fp32, _, _, _ = calculate_mfu(
        model, input_data, target, 
        nn.CrossEntropyLoss(), 
        optim.Adam(model.parameters(), lr=0.001),
        device_peak_tflops
    )
    
    # FP16
    mfu_fp16, _, _, _ = benchmark_mixed_precision(
        model, input_data, target, device_peak_tflops
    )
    
    print(f"FP32 MFU: {mfu_fp32:.2%}")
    print(f"FP16 MFU: {mfu_fp16:.2%}")
    print(f"Speedup: {mfu_fp16/mfu_fp32:.2f}x")
    
    # Plot results
    plot_mfu_results(batch_size_results, "MFU vs Batch Size")
    plot_mfu_results(model_size_results, "MFU vs Model Size")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Best batch size MFU: {max(r['mfu'] for r in batch_size_results):.2%}")
    print(f"Best model size MFU: {max(r['mfu'] for r in model_size_results):.2%}")
    print(f"Mixed precision speedup: {mfu_fp16/mfu_fp32:.2f}x")


if __name__ == "__main__":
    main()
