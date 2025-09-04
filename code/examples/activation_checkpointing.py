"""
Activation Checkpointing Example

This example demonstrates how to use activation checkpointing to reduce memory usage
during training of large models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import time
import psutil
import os


class LargeModel(nn.Module):
    """Large model that benefits from activation checkpointing"""
    
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=10, num_layers=20):
        super(LargeModel, self).__init__()
        
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
        """Standard forward pass"""
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x
    
    def forward_with_checkpointing(self, x, checkpoint_frequency=4):
        """Forward pass with activation checkpointing"""
        x = self.input_layer(x)
        
        # Create checkpoint segments
        for i in range(0, len(self.hidden_layers), checkpoint_frequency):
            segment = self.hidden_layers[i:i + checkpoint_frequency]
            x = checkpoint(self._forward_segment, x, segment)
        
        x = self.output_layer(x)
        return x
    
    def _forward_segment(self, x, segment):
        """Forward pass through a segment of layers"""
        for layer in segment:
            x = layer(x)
        return x


class CheckpointedModel(nn.Module):
    """Model with built-in activation checkpointing"""
    
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=10, num_layers=20, checkpoint_frequency=4):
        super(CheckpointedModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.checkpoint_frequency = checkpoint_frequency
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
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
        """Forward pass with automatic checkpointing"""
        x = self.input_layer(x)
        
        # Create checkpoint segments
        for i in range(0, len(self.hidden_layers), self.checkpoint_frequency):
            segment = self.hidden_layers[i:i + self.checkpoint_frequency]
            x = checkpoint(self._forward_segment, x, segment)
        
        x = self.output_layer(x)
        return x
    
    def _forward_segment(self, x, segment):
        """Forward pass through a segment of layers"""
        for layer in segment:
            x = layer(x)
        return x


def get_memory_usage():
    """Get current memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e9


def benchmark_memory_usage(model, input_data, target, criterion, optimizer, num_iterations=100):
    """Benchmark memory usage during training"""
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initial memory
    initial_memory = get_memory_usage()
    
    # Training loop
    model.train()
    for i in range(num_iterations):
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check memory every 10 iterations
        if i % 10 == 0:
            current_memory = get_memory_usage()
            print(f"Iteration {i}: Memory usage: {current_memory:.2f} GB")
    
    # Final memory
    final_memory = get_memory_usage()
    peak_memory = final_memory  # Simplified - in practice, you'd track peak
    
    return initial_memory, final_memory, peak_memory


def benchmark_training_time(model, input_data, target, criterion, optimizer, num_iterations=100):
    """Benchmark training time"""
    
    # Warmup
    for _ in range(10):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Time measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(num_iterations):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def compare_checkpointing_strategies(input_dim=1024, hidden_dim=2048, output_dim=10, num_layers=20, batch_size=32):
    """Compare different checkpointing strategies"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    input_data = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, output_dim, (batch_size,), device=device)
    
    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    print(f"Comparing checkpointing strategies on {device}")
    print(f"Model: {num_layers} layers, {hidden_dim} hidden dim, batch size {batch_size}")
    print()
    
    # Strategy 1: No checkpointing
    print("=== Strategy 1: No Checkpointing ===")
    model1 = LargeModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    
    try:
        initial_memory, final_memory, peak_memory = benchmark_memory_usage(
            model1, input_data, target, criterion, optimizer1, num_iterations=50
        )
        training_time = benchmark_training_time(
            model1, input_data, target, criterion, optimizer1, num_iterations=100
        )
        
        print(f"Memory: {initial_memory:.2f} GB -> {final_memory:.2f} GB (peak: {peak_memory:.2f} GB)")
        print(f"Training time: {training_time:.4f}s per iteration")
        print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Out of memory - cannot run without checkpointing")
            training_time = float('inf')
            peak_memory = float('inf')
        else:
            raise e
    
    # Strategy 2: Manual checkpointing
    print("\n=== Strategy 2: Manual Checkpointing ===")
    model2 = LargeModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    
    # Override forward method
    model2.forward = lambda x: model2.forward_with_checkpointing(x, checkpoint_frequency=4)
    
    try:
        initial_memory, final_memory, peak_memory = benchmark_memory_usage(
            model2, input_data, target, criterion, optimizer2, num_iterations=50
        )
        training_time = benchmark_training_time(
            model2, input_data, target, criterion, optimizer2, num_iterations=100
        )
        
        print(f"Memory: {initial_memory:.2f} GB -> {final_memory:.2f} GB (peak: {peak_memory:.2f} GB)")
        print(f"Training time: {training_time:.4f}s per iteration")
        print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Out of memory - even with checkpointing")
            training_time = float('inf')
            peak_memory = float('inf')
        else:
            raise e
    
    # Strategy 3: Built-in checkpointing
    print("\n=== Strategy 3: Built-in Checkpointing ===")
    model3 = CheckpointedModel(input_dim, hidden_dim, output_dim, num_layers, checkpoint_frequency=4).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
    
    try:
        initial_memory, final_memory, peak_memory = benchmark_memory_usage(
            model3, input_data, target, criterion, optimizer3, num_iterations=50
        )
        training_time = benchmark_training_time(
            model3, input_data, target, criterion, optimizer3, num_iterations=100
        )
        
        print(f"Memory: {initial_memory:.2f} GB -> {final_memory:.2f} GB (peak: {peak_memory:.2f} GB)")
        print(f"Training time: {training_time:.4f}s per iteration")
        print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Out of memory - even with built-in checkpointing")
            training_time = float('inf')
            peak_memory = float('inf')
        else:
            raise e


def benchmark_checkpoint_frequency(input_dim=1024, hidden_dim=2048, output_dim=10, num_layers=20, batch_size=32):
    """Benchmark different checkpoint frequencies"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    input_data = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, output_dim, (batch_size,), device=device)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    print(f"Benchmarking checkpoint frequencies on {device}")
    print(f"Model: {num_layers} layers, {hidden_dim} hidden dim, batch size {batch_size}")
    print()
    
    checkpoint_frequencies = [1, 2, 4, 8, 16, num_layers]  # num_layers means no checkpointing
    
    for freq in checkpoint_frequencies:
        print(f"=== Checkpoint Frequency: {freq} ===")
        
        if freq == num_layers:
            # No checkpointing
            model = LargeModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
        else:
            # With checkpointing
            model = CheckpointedModel(input_dim, hidden_dim, output_dim, num_layers, freq).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        try:
            # Benchmark memory
            initial_memory, final_memory, peak_memory = benchmark_memory_usage(
                model, input_data, target, criterion, optimizer, num_iterations=20
            )
            
            # Benchmark time
            training_time = benchmark_training_time(
                model, input_data, target, criterion, optimizer, num_iterations=50
            )
            
            print(f"Memory: {initial_memory:.2f} GB -> {final_memory:.2f} GB (peak: {peak_memory:.2f} GB)")
            print(f"Training time: {training_time:.4f}s per iteration")
            print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory")
            else:
                raise e
        
        print()


def get_gpu_info():
    """Get GPU information"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Running on: {device_name}")
        print(f"GPU Memory: {device_memory:.1f} GB")
        return True
    else:
        print("CUDA not available, using CPU")
        return False


def main():
    """Main function"""
    
    print("Activation Checkpointing Benchmark")
    print("=" * 50)
    
    # Check GPU setup
    is_gpu = get_gpu_info()
    print()
    
    # Configuration - adjust based on available memory
    if is_gpu:
        input_dim = 1024
        hidden_dim = 2048
        output_dim = 10
        num_layers = 20
        batch_size = 32
    else:
        # Smaller model for CPU
        input_dim = 512
        hidden_dim = 1024
        output_dim = 10
        num_layers = 10
        batch_size = 16
    
    print(f"Model configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Compare strategies
    compare_checkpointing_strategies(input_dim, hidden_dim, output_dim, num_layers, batch_size)
    
    print("\n" + "=" * 50)
    
    # Benchmark checkpoint frequencies
    benchmark_checkpoint_frequency(input_dim, hidden_dim, output_dim, num_layers, batch_size)
    
    print("\n" + "=" * 50)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
