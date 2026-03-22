# Memory Optimization Techniques

Memory optimization is crucial for training large models on limited GPU memory. This section covers the key techniques used in modern distributed training systems.

> **Note**: See `code/examples/activation_checkpointing.py` for a working benchmark of these techniques on a single GPU.

## Activation Checkpointing

### The Problem

Large models quickly run out of GPU memory due to activation storage:

```python
# Activation memory per layer (for a transformer)
activation_memory = batch_size * sequence_length * model_dim * 4  # 4 bytes per float32

# Example: batch=1, seq=4096, dim=16000
# = 1 * 4096 * 16000 * 4 = 262 MB per layer

# Total activation memory for 126 layers
total_activation_memory = 262 * 126  # ≈ 33 GB

# Plus model weights, gradients, optimizer states...
# Total memory requirement >> 80 GB (H100 memory)
```

### Solution

Instead of storing all activations, recompute them during the backward pass.

### Trade-offs

With standard PyTorch `checkpoint()` and checkpoints every C layers:

| Strategy           | Memory    | Extra Compute   | Use Case          |
| ------------------ | --------- | --------------- | ----------------- |
| Store All          | O(N)      | 0               | Small models      |
| Checkpoint Every C | O(N/C + C)| ~1 extra fwd pass| Balanced approach |
| Optimal C = √N     | O(√N)    | ~1 extra fwd pass| Recommended       |

The standard PyTorch `checkpoint()` saves intermediate activations during recomputation, so the extra compute is approximately one additional forward pass (~33% overhead), **not** O(N√N).

### PyTorch Implementation

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, layers, checkpoint_frequency=4):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_frequency = checkpoint_frequency

    def forward(self, x):
        for i in range(0, len(self.layers), self.checkpoint_frequency):
            segment = self.layers[i:i + self.checkpoint_frequency]
            x = checkpoint(self._forward_segment, x, segment,
                           use_reentrant=False)
        return x

    @staticmethod
    def _forward_segment(x, segment):
        for layer in segment:
            x = layer(x)
        return x
```

### Selective Checkpointing

Instead of uniform frequency, checkpoint based on memory pressure:

```python
def forward_with_adaptive_checkpointing(model, input_data, memory_threshold):
    activations = []
    current_memory = 0

    for i, layer in enumerate(model.layers):
        layer_memory = estimate_activation_memory(input_data)
        current_memory += layer_memory

        if current_memory > memory_threshold:
            activations.append(input_data.clone())
            current_memory = 0

        input_data = layer(input_data)

    return input_data, activations
```

## Memory-Efficient Optimizers

### Adam Memory Requirements

Standard Adam optimizer requires significant memory:

```python
# Per parameter:
# - Weights: 4 bytes (FP32) or 2 bytes (FP16)
# - Gradients: 4 bytes (FP32) or 2 bytes (FP16)
# - Momentum (beta1): 4 bytes (FP32)
# - Variance (beta2): 4 bytes (FP32)
# Total with FP32: 16 bytes per parameter
# For 1B parameter model: 1e9 * 16 = 16 GB
```

### Memory-Efficient Alternatives

#### Adafactor

```python
from transformers import Adafactor

optimizer = Adafactor(
    model.parameters(),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True
)
# Uses ~4-8 bytes/param instead of 16
```

#### 8-bit Adam

```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999)
)
# Uses ~10 bytes/param instead of 16
```

## Gradient Accumulation

### Concept

Accumulate gradients over multiple micro-batches before updating weights:

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory Benefits**:

- Reduces peak activation memory by factor of `accumulation_steps`
- Enables larger effective batch sizes without more memory
- Maintains training stability

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

```python
scaler = torch.amp.GradScaler('cuda')

for batch in dataloader:
    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        outputs = model(batch)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Memory Savings**:

- **FP16/BF16**: 2x less memory than FP32 for activations
- **INT8**: 4x less memory than FP32 (quantization)

## CPU Offloading

### Offload Parameters to CPU

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision

fsdp_model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
)
```

### Offload Optimizer States

```python
def offload_optimizer_to_cpu(model, optimizer):
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad.cpu()

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cpu()
```

## Memory Profiling

### PyTorch Memory Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

### Memory Monitoring

```python
def print_memory_stats():
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Max Allocated: {max_allocated:.2f} GB")
```

## Memory-Efficient Data Loading

### Streaming Data Loading

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### Lazy Loading

```python
class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __getitem__(self, idx):
        return torch.load(self.data_paths[idx])

    def __len__(self):
        return len(self.data_paths)
```

## Memory Optimization Checklist

### Before Training

- [ ] Profile baseline memory usage
- [ ] Choose appropriate optimizer (Adam vs Adafactor vs 8-bit Adam)
- [ ] Set up mixed precision training (AMP)
- [ ] Configure activation checkpointing
- [ ] Set up CPU offloading if needed

### During Training

- [ ] Monitor memory usage continuously
- [ ] Adjust batch size based on memory
- [ ] Use gradient accumulation if needed
- [ ] Implement memory-efficient data loading

### After Training

- [ ] Analyze memory usage patterns
- [ ] Optimize for future runs
- [ ] Document memory requirements

## Best Practices

1. **Start Simple**: Begin with basic optimizations (AMP first)
2. **Profile First**: Measure before optimizing
3. **Incremental**: Add optimizations one at a time
4. **Test Thoroughly**: Each optimization can affect convergence
5. **Monitor Continuously**: Memory usage can change during training

## Common Pitfalls

1. **Over-optimization**: Too many optimizations can hurt performance
2. **Incorrect Scaling**: Wrong scaling factors in mixed precision
3. **Memory Leaks**: Not properly cleaning up tensors
4. **Synchronization Issues**: Incorrect gradient synchronization in distributed settings
5. **Hardware Limitations**: Not considering GPU memory limits before choosing batch size
