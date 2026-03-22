"""
Activation Checkpointing Example

This example demonstrates how to use activation checkpointing to reduce memory usage
during training of large models on a single GPU.
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
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Standard forward pass (no checkpointing)"""
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x

    def forward_with_checkpointing(self, x, checkpoint_frequency=4):
        """Forward pass with activation checkpointing"""
        x = self.input_layer(x)

        for i in range(0, len(self.hidden_layers), checkpoint_frequency):
            segment = self.hidden_layers[i:i + checkpoint_frequency]
            x = checkpoint(
                self._forward_segment, x, segment,
                use_reentrant=False
            )

        x = self.output_layer(x)
        return x

    @staticmethod
    def _forward_segment(x, segment):
        for layer in segment:
            x = layer(x)
        return x


class CheckpointedModel(nn.Module):
    """Model with built-in activation checkpointing"""

    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=10,
                 num_layers=20, checkpoint_frequency=4):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.checkpoint_frequency = checkpoint_frequency

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with automatic checkpointing"""
        x = self.input_layer(x)

        for i in range(0, len(self.hidden_layers), self.checkpoint_frequency):
            segment = self.hidden_layers[i:i + self.checkpoint_frequency]
            x = checkpoint(
                self._forward_segment, x, segment,
                use_reentrant=False
            )

        x = self.output_layer(x)
        return x

    @staticmethod
    def _forward_segment(x, segment):
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


def get_peak_memory():
    """Get peak GPU memory allocated in GB (since last reset)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def benchmark_memory_usage(model, input_data, target, criterion, optimizer,
                           num_iterations=100):
    """Benchmark memory usage during training, tracking true peak."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    initial_memory = get_memory_usage()

    model.train()
    for i in range(num_iterations):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            current_memory = get_memory_usage()
            print(f"Iteration {i}: Memory usage: {current_memory:.2f} GB")

    final_memory = get_memory_usage()
    peak_memory = get_peak_memory() if torch.cuda.is_available() else final_memory

    return initial_memory, final_memory, peak_memory


def benchmark_training_time(model, input_data, target, criterion, optimizer,
                            num_iterations=100):
    """Benchmark training time"""
    for _ in range(10):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

    return (end_time - start_time) / num_iterations


def _run_strategy(name, model, input_data, target, criterion, device):
    """Run a single checkpointing strategy benchmark in isolation."""
    print(f"\n=== {name} ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    try:
        initial_memory, final_memory, peak_memory = benchmark_memory_usage(
            model, input_data, target, criterion, optimizer, num_iterations=50
        )
        training_time = benchmark_training_time(
            model, input_data, target, criterion, optimizer, num_iterations=100
        )

        print(f"Memory: {initial_memory:.2f} GB -> {final_memory:.2f} GB "
              f"(peak: {peak_memory:.2f} GB)")
        print(f"Training time: {training_time:.4f}s per iteration")
        print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
        return peak_memory, training_time

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Out of memory for {name}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float('inf'), float('inf')
        raise


def compare_checkpointing_strategies(input_dim=1024, hidden_dim=2048,
                                     output_dim=10, num_layers=20,
                                     batch_size=32):
    """Compare different checkpointing strategies with proper isolation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_data = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, output_dim, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    print(f"Comparing checkpointing strategies on {device}")
    print(f"Model: {num_layers} layers, {hidden_dim} hidden dim, batch size {batch_size}")

    model1 = LargeModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    peak1, time1 = _run_strategy(
        "Strategy 1: No Checkpointing",
        model1, input_data, target, criterion, device
    )
    del model1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model2 = LargeModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    model2.forward = lambda x: model2.forward_with_checkpointing(x, checkpoint_frequency=4)
    peak2, time2 = _run_strategy(
        "Strategy 2: Manual Checkpointing (freq=4)",
        model2, input_data, target, criterion, device
    )
    del model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model3 = CheckpointedModel(input_dim, hidden_dim, output_dim, num_layers,
                               checkpoint_frequency=4).to(device)
    peak3, time3 = _run_strategy(
        "Strategy 3: Built-in Checkpointing (freq=4)",
        model3, input_data, target, criterion, device
    )
    del model3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Comparison Summary ===")
    print(f"{'Strategy':<35} {'Peak Memory':>12} {'Time/iter':>12}")
    print("-" * 60)
    for label, peak, t in [
        ("No Checkpointing", peak1, time1),
        ("Manual Checkpointing (freq=4)", peak2, time2),
        ("Built-in Checkpointing (freq=4)", peak3, time3),
    ]:
        peak_str = f"{peak:.2f} GB" if peak != float('inf') else "OOM"
        time_str = f"{t:.4f}s" if t != float('inf') else "N/A"
        print(f"{label:<35} {peak_str:>12} {time_str:>12}")


def benchmark_checkpoint_frequency(input_dim=1024, hidden_dim=2048,
                                   output_dim=10, num_layers=20,
                                   batch_size=32):
    """Benchmark different checkpoint frequencies with proper isolation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_data = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, output_dim, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    print(f"\nBenchmarking checkpoint frequencies on {device}")
    print(f"Model: {num_layers} layers, {hidden_dim} hidden dim, batch size {batch_size}")

    frequencies = [1, 2, 4, 8, 16]
    results = []

    for freq in frequencies:
        model = CheckpointedModel(input_dim, hidden_dim, output_dim,
                                  num_layers, freq).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            initial_memory, final_memory, peak_memory = benchmark_memory_usage(
                model, input_data, target, criterion, optimizer, num_iterations=20
            )
            training_time = benchmark_training_time(
                model, input_data, target, criterion, optimizer, num_iterations=50
            )

            results.append({
                'frequency': freq,
                'peak_memory': peak_memory,
                'training_time': training_time,
            })

            print(f"\nFrequency {freq}: peak={peak_memory:.2f} GB, "
                  f"time={training_time:.4f}s/iter")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nFrequency {freq}: Out of memory")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
        finally:
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    no_ckpt_model = LargeModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    no_ckpt_optimizer = optim.Adam(no_ckpt_model.parameters(), lr=0.001)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        _, _, peak = benchmark_memory_usage(
            no_ckpt_model, input_data, target, criterion, no_ckpt_optimizer,
            num_iterations=20
        )
        t = benchmark_training_time(
            no_ckpt_model, input_data, target, criterion, no_ckpt_optimizer,
            num_iterations=50
        )
        results.append({'frequency': 'none', 'peak_memory': peak, 'training_time': t})
        print(f"\nNo checkpointing: peak={peak:.2f} GB, time={t:.4f}s/iter")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nNo checkpointing: Out of memory")
        else:
            raise
    finally:
        del no_ckpt_model, no_ckpt_optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if results:
        print("\n=== Checkpoint Frequency Summary ===")
        print(f"{'Frequency':<15} {'Peak Memory':>12} {'Time/iter':>12}")
        print("-" * 40)
        for r in results:
            freq_label = str(r['frequency'])
            peak_str = f"{r['peak_memory']:.2f} GB"
            time_str = f"{r['training_time']:.4f}s"
            print(f"{freq_label:<15} {peak_str:>12} {time_str:>12}")


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

    is_gpu = get_gpu_info()
    print()

    if is_gpu:
        input_dim = 1024
        hidden_dim = 2048
        output_dim = 10
        num_layers = 20
        batch_size = 32
    else:
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

    compare_checkpointing_strategies(input_dim, hidden_dim, output_dim,
                                     num_layers, batch_size)

    print("\n" + "=" * 50)

    benchmark_checkpoint_frequency(input_dim, hidden_dim, output_dim,
                                   num_layers, batch_size)

    print("\n" + "=" * 50)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
