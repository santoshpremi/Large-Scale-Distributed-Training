# Performance Optimization

Performance optimization is crucial for efficient distributed training. This section covers the key metrics, techniques, and tools for optimizing training performance.

> **Note**: See `code/examples/mfu_benchmarking.py` for a working single-GPU MFU benchmark.

## Model Flops Utilization (MFU)

### Definition

MFU measures what fraction of the GPU's theoretical peak throughput is being used for actual model computation.

```
MFU = (model FLOPS per step / step time) / device peak FLOPS/s
```

### Why MFU Matters

- **Efficiency Metric**: Measures how well you're using available compute
- **Optimization Target**: Higher MFU = better performance
- **Bottleneck Identification**: Low MFU indicates bottlenecks
- **Scaling Validation**: Ensures performance scales with hardware

### Calculating MFU

#### Step 1: Count Model FLOPS

For Linear layers, FLOPS = 2 × batch_size × in_features × out_features (multiply-add counted separately):

```python
def linear_flops(batch_size, in_features, out_features):
    return 2 * batch_size * in_features * out_features

def model_flops_per_step(model, batch_size):
    """Total FLOPS for forward + backward (≈3x forward)."""
    forward_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            forward_flops += linear_flops(batch_size,
                                          module.in_features,
                                          module.out_features)
    return 3 * forward_flops  # forward + grad_input + grad_weight
```

#### Step 2: Measure Step Time

```python
torch.cuda.synchronize()
start = time.time()

output = model(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

torch.cuda.synchronize()
step_time = time.time() - start
```

#### Step 3: Compute MFU

```python
achieved_flops_per_sec = model_flops / step_time
mfu = achieved_flops_per_sec / (device_peak_tflops * 1e12)
```

**Important**: Use the correct peak for your precision mode. FP32 workloads should reference the FP32 peak (e.g. 67 TFLOPS for H100), not the tensor-core FP16 peak (989 TFLOPS).

### MFU Targets

| MFU Range | Performance Level | Action Required                 |
| --------- | ----------------- | ------------------------------- |
| < 20%     | Poor              | Major optimization needed       |
| 20–30%    | Below Average     | Significant optimization needed |
| 30–40%    | Good              | Minor optimizations             |
| 40–50%    | Excellent         | Well optimized                  |
| > 50%     | Outstanding       | Near optimal                    |

## Hardware Flops Utilization (HFU)

### Definition

HFU measures the fraction of theoretical GPU throughput achieved for raw computation (e.g. a large matrix multiply with no other overhead).

### HFU vs MFU

| Metric | What It Measures               | Use Case              |
| ------ | ------------------------------ | --------------------- |
| HFU    | Raw computation efficiency     | Hardware optimization |
| MFU    | End-to-end training efficiency | System optimization   |

### HFU Benchmarking

```python
def benchmark_hfu(device, peak_tflops):
    size = 8192
    A = torch.randn(size, size, device=device, dtype=torch.float16)
    B = torch.randn(size, size, device=device, dtype=torch.float16)

    for _ in range(10):  # warmup
        torch.mm(A, B)

    torch.cuda.synchronize()
    start = time.time()

    iterations = 100
    for _ in range(iterations):
        torch.mm(A, B)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    flops_per_iter = 2 * size ** 3
    achieved = (flops_per_iter * iterations) / elapsed
    hfu = achieved / (peak_tflops * 1e12)
    print(f"HFU: {hfu:.2%}")
    return hfu
```

## Communication Optimization

### Communication Patterns

#### All-Reduce

Used in data parallelism to average gradients:

```python
import torch.distributed as dist

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
tensor /= world_size
```

Effective bandwidth for ring all-reduce: `2 * (N-1)/N * data_size / time`.

#### Broadcast

Used in FSDP to distribute gathered weights:

```python
dist.broadcast(tensor, src=0)
```

### Communication Overlap

Overlap gradient communication with the next forward pass:

```python
# During backward, start async gradient all-reduce as soon as
# each layer's gradients are ready (this is what DDP does internally).
for param in model.parameters():
    if param.grad is not None:
        handle = dist.all_reduce(param.grad, async_op=True)
        # Continue computation while communication happens
        handles.append(handle)

# Wait for all communications to complete before optimizer step
for handle in handles:
    handle.wait()
```

## Batch Size Optimization

### Finding Optimal Batch Size

```python
def find_optimal_batch_size(model, device, peak_tflops):
    batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    best_mfu, best_bs = 0, 8

    for bs in batch_sizes:
        try:
            data = torch.randn(bs, input_dim, device=device)
            target = torch.randint(0, num_classes, (bs,), device=device)
            mfu = measure_mfu(model, data, target, peak_tflops)
            if mfu > best_mfu:
                best_mfu, best_bs = mfu, bs
        except RuntimeError:  # OOM
            torch.cuda.empty_cache()
            break

    return best_bs
```

### Dynamic Batch Size

Increase batch size during training to improve utilization:

```python
current_batch_size = initial_batch_size

for epoch in range(num_epochs):
    for batch in dataloader:
        try:
            batch_data = batch[:current_batch_size]
            output = model(batch_data)
            loss = criterion(output, target)
            loss.backward()

            # Increase batch size gradually
            current_batch_size = min(int(current_batch_size * 1.1),
                                     max_batch_size)
        except RuntimeError:  # OOM
            current_batch_size = max(int(current_batch_size * 0.8), 1)
            torch.cuda.empty_cache()
```

## Learning Rate Optimization

### Warmup + Cosine Decay

```python
import math

def get_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, get_lr_lambda(warmup_steps=1000, total_steps=100000)
)
```

## Profiling and Monitoring

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        output = model(batch[0])
        loss = criterion(output, batch[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Real-time Monitoring

```python
def monitor_training(model, dataloader, device, peak_tflops, interval=50):
    mfu_history = []
    memory_history = []

    for i, batch in enumerate(dataloader):
        output = model(batch[0].to(device))
        loss = criterion(output, batch[1].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % interval == 0:
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                memory_history.append(mem)

            if len(mfu_history) > 10:
                recent = sum(mfu_history[-10:]) / 10
                if recent < 0.2:
                    print(f"Warning: Low MFU ({recent:.1%}) at step {i}")
```

## Optimization Checklist

- [ ] Measure baseline MFU
- [ ] Profile bottlenecks (compute vs memory vs communication)
- [ ] Optimize batch size
- [ ] Enable mixed precision
- [ ] Optimize communication (overlap, compression)
- [ ] Monitor continuously
- [ ] Test for regressions

## Common Pitfalls

1. **Premature Optimization**: Always profile before optimizing
2. **Wrong Peak Reference**: Compare FP32 workloads against FP32 peak, not tensor-core peak
3. **Ignoring Communication**: Communication often dominates at scale
4. **Memory Fragmentation**: OOM doesn't always mean you need less memory
5. **Synchronization Overhead**: Unnecessary `torch.cuda.synchronize()` calls hurt throughput

## Performance Targets

- **MFU**: > 30% (good), > 40% (excellent)
- **Memory Usage**: < 80% of GPU memory (leave headroom)
- **Communication**: < 20% of total step time
- **Scalability**: Near-linear scaling with GPUs
