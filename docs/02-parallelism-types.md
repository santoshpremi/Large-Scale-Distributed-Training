# Six Types of Parallelism in Distributed Training

Understanding the different types of parallelism is crucial for designing efficient distributed training systems. Each type has its own characteristics, use cases, and communication patterns.

> **Note**: The code examples in this repository focus on single-GPU fundamentals. This document describes the distributed concepts that build on those fundamentals.

## 1. Data Parallelism (DP)

### Concept

Split the batch across GPUs, where each GPU processes different data samples but maintains a full copy of the model.

### How It Works

```
GPU 0: model_copy + batch[0:B/N]  ──┐
GPU 1: model_copy + batch[B/N:2B/N] ──┤── All-reduce gradients ── Update weights
GPU 2: model_copy + batch[2B/N:3B/N] ──┤
GPU N: model_copy + batch[...:]    ──┘
```

### Key Characteristics

- **Memory**: Each GPU stores full model (~16 bytes per parameter with Adam)
- **Communication**: All-reduce of gradients after each backward pass
- **Scaling**: Limited by model size, not batch size
- **Best For**: Models that fit on single GPU

### Mathematical Foundation

The gradient computation can be parallelized because gradients are linear:

```
∇L(θ) = (1/B) Σᵢ ∇ℓ(θ, xᵢ)
       = (1/N) Σⱼ [(1/B_local) Σᵢ∈partⱼ ∇ℓ(θ, xᵢ)]
```

Each GPU computes the inner sum over its local batch, then all-reduce averages.

### PyTorch Implementation

```python
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### Advantages

- Simple to implement
- No model architecture changes needed
- Good for smaller models
- Automatic gradient synchronization

### Disadvantages

- Memory limited by single GPU
- Communication scales with model size
- Not suitable for very large models

## 2. Fully Sharded Data Parallelism (FSDP)

### Concept

Split model weights across GPUs, not just data. Each GPU owns a portion of the model weights.

### How It Works

```
Forward pass:
  For each layer:
    1. All-gather weights from all GPUs     ← communication
    2. Compute forward with full weights
    3. Discard non-owned weight shards       ← save memory

Backward pass:
  For each layer (reverse):
    1. All-gather weights                    ← communication
    2. Compute gradients
    3. Reduce-scatter gradients              ← communication
    4. Discard non-owned weight shards
```

### Key Characteristics

- **Memory**: Model weights distributed across GPUs
- **Communication**: 3x model size per forward/backward pass
- **Scaling**: Limited by activation memory
- **Best For**: Large models that don't fit on single GPU

### Memory Savings Example

For a 1B parameter model with mixed-precision Adam:

| Component         | Per-param bytes | 1B params total |
| ----------------- | --------------- | --------------- |
| FP16 weights      | 2               | 2 GB            |
| FP16 gradients    | 2               | 2 GB            |
| FP32 master weights | 4             | 4 GB            |
| FP32 momentum     | 4               | 4 GB            |
| FP32 variance     | 4               | 4 GB            |
| **Total**         | **16**          | **16 GB**       |

With FSDP across 8 GPUs: ~2 GB sharded state per GPU + activation memory.

### PyTorch Implementation

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

### Advantages

- Memory efficient for large models
- Can train models larger than single GPU memory
- Automatic weight sharding and gathering

### Disadvantages

- More complex communication patterns
- Higher communication overhead
- Requires careful memory management

## 3. Hybrid Sharded Data Parallelism (HSDP)

### Concept

Two-dimensional parallelism combining FSDP and DP. Groups of GPUs do FSDP internally, groups do DP with each other.

### How It Works

```
Shard Group A (FSDP):  GPU 0, 1, 2, 3  ──┐
                                           ├── DP across groups (gradient all-reduce)
Shard Group B (FSDP):  GPU 4, 5, 6, 7  ──┘
```

### Key Characteristics

- **Memory**: FSDP within groups, DP across groups
- **Communication**: Intra-group sharding + inter-group gradient sync
- **Scaling**: Optimizes for cluster topology
- **Best For**: Large clusters with hierarchical interconnect

### PyTorch Implementation

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)
```

### Advantages

- Optimizes for cluster topology
- Different communication patterns for different levels
- Can scale to very large clusters

### Disadvantages

- Complex setup and configuration
- Requires understanding of cluster topology
- More difficult to debug

## 4. Context Parallelism

### Concept

Split sequences across GPUs for very long contexts. Each GPU processes different parts of the same sequence.

### How It Works

```
Sequence: [tok_1, tok_2, ..., tok_128K]

GPU 0: [tok_1 ... tok_32K]    ──┐
GPU 1: [tok_32K+1 ... tok_64K] ──┤── Communicate KV blocks for attention
GPU 2: [tok_64K+1 ... tok_96K] ──┤
GPU 3: [tok_96K+1 ... tok_128K]──┘
```

### Key Characteristics

- **Memory**: Activations distributed across GPUs
- **Communication**: Attention computation requires special handling
- **Scaling**: Limited by attention computation complexity
- **Best For**: Very long sequences (>10K tokens)

### Attention Parallelization Strategies

#### Ring Attention

Each GPU holds its own Q chunk and circulates K/V blocks around a ring:

```
Step 1: GPU_i computes attn(Q_i, K_i, V_i)
Step 2: GPU_i receives K_{i-1}, V_{i-1}, computes partial attention
...
Step N: All partial attentions accumulated
```

#### Ulysses Attention

Split along the head dimension instead of the sequence dimension, then use all-to-all communication to redistribute.

### Advantages

- Enables very long sequences
- Memory efficient for long contexts
- Can process sequences longer than single GPU memory

### Disadvantages

- Complex attention computation
- Communication overhead for attention
- Limited by attention complexity

## 5. Pipeline Parallelism

### Concept

Split model layers across GPUs. Each GPU processes different layers of the model.

### How It Works

```
GPU 0: Layers 0-3   (stage 0)
GPU 1: Layers 4-7   (stage 1)
GPU 2: Layers 8-11  (stage 2)
GPU 3: Layers 12-15 (stage 3)

Timeline with 4 microbatches (F=forward, B=backward):
GPU 0: F0 F1 F2 F3 __ __ __ B3 B2 B1 B0
GPU 1: __ F0 F1 F2 F3 __ B3 B2 B1 B0 __
GPU 2: __ __ F0 F1 F2 F3 B3 B2 B1 __ __
GPU 3: __ __ __ F0 F1 F2 B3 B2 __ __ __
                        ↑ bubble time ↑
```

### Key Characteristics

- **Memory**: Each GPU stores only assigned layers
- **Communication**: Sequential dependencies between stages
- **Bubble fraction**: (N-1) / (M+N-1) for N stages, M microbatches
- **Efficiency**: M / (M+N-1), approaches 1 as M increases

### Advantages

- Memory efficient for deep models
- Each GPU processes only assigned layers
- Can handle very deep models

### Disadvantages

- Sequential dependencies limit parallelism
- Bubble time reduces efficiency
- Complex scheduling required

## 6. Tensor Parallelism

### Concept

Split individual weight matrices across GPUs. Each GPU computes a portion of the matrix multiplication.

### How It Works

For a linear layer `Y = XW + b`:

```
W split column-wise:  W = [W_0 | W_1 | ... | W_{N-1}]

GPU_i computes:  Y_i = X @ W_i    (local matmul, no communication)
Then:            Y = concat(Y_0, Y_1, ...) or all-reduce depending on next layer
```

### Two-Layer Trick

For consecutive MLP layers (common in transformers), split W1 by columns and W2 by rows to avoid intermediate communication:

```python
# Split W1 by columns, W2 by rows
W1_chunks = W1.chunk(num_gpus, dim=1)  # Column-wise split
W2_chunks = W2.chunk(num_gpus, dim=0)  # Row-wise split

# Each GPU computes: x @ W1_chunk @ W2_chunk
intermediate = x @ W1_chunks[rank]
output_local = intermediate @ W2_chunks[rank]

# Only need one all-reduce at the end
output = all_reduce(output_local)
```

### Key Characteristics

- **Memory**: Weight matrices split across GPUs
- **Communication**: Gather activations after each layer (or layer pair)
- **Scaling**: Limited by matrix dimensions
- **Best For**: Wide models with large weight matrices

### Advantages

- Memory efficient for wide models
- Good for large weight matrices
- Works well with transformer MLPs

### Disadvantages

- Communication after each layer
- Limited by matrix dimensions
- Complex for non-linear layers

## Multi-Dimensional Parallelism

### Combining All Types

In practice, modern systems use multiple types of parallelism simultaneously:

```python
# Llama3-405B configuration (approximate)
config = {
    'tensor_parallelism': 8,    # 8-way TP within a server
    'pipeline_parallelism': 16, # 16-way PP across servers
    'data_parallelism': 128,    # 128-way DP across the cluster
}
# Total: 8 × 16 × 128 = 16,384 GPUs
```

### Choosing the Right Parallelism

#### Decision Tree

1. Model fits on one GPU → **Data Parallelism**
2. Model doesn't fit → **FSDP** (or HSDP for >256 GPUs)
3. Very long sequences → add **Context Parallelism**
4. Very deep model → add **Pipeline Parallelism**
5. Very wide layers → add **Tensor Parallelism**

### Performance Considerations

1. **Memory**: FSDP for large models
2. **Communication**: Optimize for cluster topology (TP intra-node, PP inter-node)
3. **Sequences**: Context parallelism for long sequences
4. **Depth**: Pipeline parallelism for deep models
5. **Width**: Tensor parallelism for wide models

### Best Practices

1. **Start Simple**: Begin with data parallelism
2. **Profile First**: Measure MFU before optimizing
3. **Incremental**: Add parallelism types one at a time
4. **Test Thoroughly**: Each type adds complexity
5. **Monitor Performance**: Track MFU and communication overhead
