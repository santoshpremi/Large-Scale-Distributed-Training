"""
Model Flops Utilization (MFU) Benchmarking

This example demonstrates how to benchmark and optimize Model Flops Utilization
on a single GPU setup (T4/P100/V100).
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class BenchmarkModel(nn.Module):
    """Model for MFU benchmarking"""

    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=10, num_layers=10):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
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
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x


def _linear_flops(batch_size, in_features, out_features):
    """FLOPS for a single Linear layer: 2 * B * I * O (multiply-add counted separately)."""
    return 2 * batch_size * in_features * out_features


def calculate_model_flops(model, batch_size):
    """Calculate theoretical FLOPS for one forward + backward pass.

    Uses the standard convention: backward ≈ 2x forward FLOPS,
    so total = 3 * forward FLOPS.
    """
    forward_flops = 0

    forward_flops += _linear_flops(batch_size, model.input_layer.in_features,
                                   model.input_layer.out_features)

    for layer in model.hidden_layers:
        linear = layer[0]
        forward_flops += _linear_flops(batch_size, linear.in_features,
                                       linear.out_features)

    forward_flops += _linear_flops(batch_size, model.output_layer.in_features,
                                   model.output_layer.out_features)

    total_flops = 3 * forward_flops
    return total_flops


def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure_training_time(model, input_data, target, criterion, optimizer,
                          device, num_iterations=100):
    """Measure average wall-clock time per training step."""
    model.train()

    for _ in range(10):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    _sync_device(device)
    start_time = time.time()

    for _ in range(num_iterations):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    _sync_device(device)
    end_time = time.time()

    return (end_time - start_time) / num_iterations


def calculate_mfu(model, input_data, target, criterion, optimizer,
                  device_peak_tflops, device):
    """Calculate Model Flops Utilization.

    MFU = (model FLOPS per step / step time) / device peak FLOPS/s
    """
    batch_size = input_data.size(0)
    theoretical_flops = calculate_model_flops(model, batch_size)

    theoretical_time = theoretical_flops / (device_peak_tflops * 1e12)

    actual_time = measure_training_time(
        model, input_data, target, criterion, optimizer, device)

    mfu = theoretical_time / actual_time

    return mfu, theoretical_flops, theoretical_time, actual_time


def benchmark_mfu_vs_batch_size(base_model, device, device_peak_tflops,
                                max_batch_size=1024):
    """Benchmark MFU vs batch size using fresh model copies for each test."""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    mfu_results = []
    initial_state = copy.deepcopy(base_model.state_dict())

    for batch_size in batch_sizes:
        if batch_size > max_batch_size:
            break

        try:
            base_model.load_state_dict(initial_state)

            input_data = torch.randn(batch_size, base_model.input_layer.in_features,
                                     device=device)
            target = torch.randint(0, base_model.output_layer.out_features,
                                   (batch_size,), device=device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(base_model.parameters(), lr=0.001)

            mfu, theoretical_flops, theoretical_time, actual_time = calculate_mfu(
                base_model, input_data, target, criterion, optimizer,
                device_peak_tflops, device
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
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            raise

    base_model.load_state_dict(initial_state)
    return mfu_results


def benchmark_mfu_vs_model_size(device, device_peak_tflops, batch_size=32):
    """Benchmark MFU vs model size (each config gets a fresh model)."""
    hidden_dims = [512, 1024, 2048, 4096, 8192]
    num_layers_list = [5, 10, 20, 40]
    mfu_results = []

    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            try:
                model = BenchmarkModel(
                    input_dim=1024,
                    hidden_dim=hidden_dim,
                    output_dim=10,
                    num_layers=num_layers
                ).to(device)

                input_data = torch.randn(batch_size, 1024, device=device)
                target = torch.randint(0, 10, (batch_size,), device=device)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                mfu, theoretical_flops, theoretical_time, actual_time = calculate_mfu(
                    model, input_data, target, criterion, optimizer,
                    device_peak_tflops, device
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

                del model, optimizer, input_data, target
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Hidden dim {hidden_dim}, Layers {num_layers}: Out of memory")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    break
                raise

    return mfu_results


def benchmark_mixed_precision(model, input_data, target, device_peak_tflops, device):
    """Benchmark MFU with mixed precision (requires CUDA)."""
    if device.type != 'cuda':
        print("Mixed precision benchmark requires CUDA, skipping.")
        return None, 0, 0, 0

    initial_state = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')

    for _ in range(10):
        with torch.amp.autocast('cuda'):
            output = model(input_data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    start_time = time.time()

    num_iterations = 100
    for _ in range(num_iterations):
        with torch.amp.autocast('cuda'):
            output = model(input_data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    end_time = time.time()

    batch_size = input_data.size(0)
    theoretical_flops = calculate_model_flops(model, batch_size)
    theoretical_time = theoretical_flops / (device_peak_tflops * 1e12)
    actual_time = (end_time - start_time) / num_iterations

    mfu = theoretical_time / actual_time

    model.load_state_dict(initial_state)

    return mfu, theoretical_flops, theoretical_time, actual_time


def plot_mfu_results(mfu_results, title="MFU Benchmark Results"):
    """Plot MFU benchmark results."""
    if not mfu_results:
        print("No results to plot")
        return

    if 'batch_size' in mfu_results[0]:
        x_vals = [r['batch_size'] for r in mfu_results]
        x_label = 'Batch Size'
        marker = 'bo-'
    elif 'hidden_dim' in mfu_results[0]:
        x_vals = [r['hidden_dim'] for r in mfu_results]
        x_label = 'Hidden Dimension'
        marker = 'ro-'
    else:
        return

    mfus = [r['mfu'] for r in mfu_results]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, mfus, marker)
    plt.xlabel(x_label)
    plt.ylabel('Model Flops Utilization (MFU)')
    plt.title(title)
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


def get_device_specs():
    """Get device specifications for single GPU setup.

    Returns (device, fp32_peak_tflops, fp16_tensor_peak_tflops).
    The two peaks allow choosing the right reference for each precision mode.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        if "T4" in device_name:
            fp32_peak, fp16_peak = 8.1, 65.1
        elif "P100" in device_name:
            fp32_peak, fp16_peak = 10.6, 21.2
        elif "V100" in device_name:
            fp32_peak, fp16_peak = 15.7, 125.0
        elif "A100" in device_name:
            fp32_peak, fp16_peak = 19.5, 312.0
        elif "H100" in device_name:
            fp32_peak, fp16_peak = 66.9, 989.4
        else:
            fp32_peak, fp16_peak = 8.1, 65.1

        print(f"GPU: {device_name}")
        print(f"Memory: {device_memory:.1f} GB")
        print(f"FP32 Peak: {fp32_peak} TFLOPs | FP16 Tensor Peak: {fp16_peak} TFLOPs")

        return torch.device('cuda'), fp32_peak, fp16_peak
    else:
        print("CUDA not available, using CPU (MFU numbers will be approximate)")
        return torch.device('cpu'), 0.1, 0.1


def main():
    """Main benchmarking function"""
    device, fp32_peak, fp16_peak = get_device_specs()

    print(f"\nRunning MFU benchmarks on {device}")

    model = BenchmarkModel(
        input_dim=1024,
        hidden_dim=2048,
        output_dim=10,
        num_layers=10
    ).to(device)

    print("\n=== Benchmark 1: MFU vs Batch Size (FP32) ===")
    batch_size_results = benchmark_mfu_vs_batch_size(
        model, device, fp32_peak)

    print("\n=== Benchmark 2: MFU vs Model Size (FP32) ===")
    model_size_results = benchmark_mfu_vs_model_size(
        device, fp32_peak)

    print("\n=== Benchmark 3: FP32 vs Mixed Precision ===")
    input_data = torch.randn(32, 1024, device=device)
    target = torch.randint(0, 10, (32,), device=device)

    initial_state = copy.deepcopy(model.state_dict())

    mfu_fp32, _, _, _ = calculate_mfu(
        model, input_data, target,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        fp32_peak, device
    )
    model.load_state_dict(initial_state)

    mfu_fp16, _, _, _ = benchmark_mixed_precision(
        model, input_data, target, fp16_peak, device
    )

    print(f"FP32 MFU (vs FP32 peak): {mfu_fp32:.2%}")
    if mfu_fp16 is not None:
        print(f"FP16 MFU (vs FP16 tensor peak): {mfu_fp16:.2%}")
        print(f"Speedup: {mfu_fp16/mfu_fp32:.2f}x")
    else:
        print("FP16 benchmark skipped (no CUDA)")

    if batch_size_results:
        plot_mfu_results(batch_size_results, "MFU vs Batch Size")
    if model_size_results:
        plot_mfu_results(model_size_results, "MFU vs Model Size")

    print("\n=== Summary ===")
    if batch_size_results:
        print(f"Best batch size MFU: {max(r['mfu'] for r in batch_size_results):.2%}")
    if model_size_results:
        print(f"Best model size MFU: {max(r['mfu'] for r in model_size_results):.2%}")
    if mfu_fp16 is not None:
        print(f"Mixed precision speedup: {mfu_fp16/mfu_fp32:.2f}x")


if __name__ == "__main__":
    main()
