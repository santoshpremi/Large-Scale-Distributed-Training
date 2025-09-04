# Large Scale Distributed Training

A comprehensive guide to large-scale distributed training covering GPU architecture, parallelism techniques, and optimization strategies.

## Table of Contents

1. [Overview](#overview)
2. [Five Types of Parallelism](#five-types-of-parallelism)
3. [Memory Optimization Techniques](#memory-optimization-techniques)
4. [Performance Optimization](#performance-optimization)
5. [GPU Hardware Deep Dive](#gpu-hardware-deep-dive)
6. [GPU Clusters and Infrastructure](#gpu-clusters-and-infrastructure)
7. [Resources](#resources)
8. [Code Examples](#code-examples)

## Overview

This repository contains a comprehensive guide to large-scale distributed training, covering everything from individual GPU architecture to training models on tens of thousands of GPUs. The content draws from multiple sources including academic papers, industry research, and course materials, with a focus on practical implementation using PyTorch.

## Single GPU Compatible

All code examples are optimized for single GPU setups (T4/P100/V100). The experiments demonstrate key concepts in distributed training using techniques that work within single GPU constraints.

### Key Topics Covered

- **GPU Architecture**: Deep dive into H100, memory hierarchy, and tensor cores
- **Distributed Training**: Five different types of parallelism
- **Memory Management**: Activation checkpointing and sharding strategies
- **Performance Optimization**: Model Flops Utilization (MFU) and benchmarking
- **Real-world Examples**: Llama3-405B training case study

### Available Experiments (Single GPU Ready)

1. **Single GPU Training** (`single_gpu_training.py`)

   - Mixed precision training with CIFAR-10
   - Performance benchmarking and visualization
   - Memory optimization techniques

2. **MFU Benchmarking** (`mfu_benchmarking.py`)

   - Model Flops Utilization analysis
   - Batch size optimization
   - Mixed precision performance comparison

3. **Activation Checkpointing** (`activation_checkpointing.py`)
   - Memory vs compute trade-offs
   - Different checkpointing strategies
   - Performance impact analysis

### Why This Matters

> "We've been living through a 1,000x increase in computation over the past 12 years. Anytime anything in the world changes by 1,000x, you should step up and pay attention because that's going to cause major changes in our technological capabilities."

- **2013**: K40 GPU - 5 TFLOPs
- **2025**: B200 GPU - 5,000 TFLOPs (tensor cores)
- **Scale**: From single GPU to 24,000+ GPU clusters

## Five Types of Parallelism

### 1. Data Parallelism (DP)

**Concept**: Split the batch across GPUs, each GPU processes different data samples.

**Key Points**:

- Each GPU maintains a full copy of the model
- Gradients are averaged across all GPUs
- Communication: All-reduce of gradients
- Best for: Models that fit on single GPU

**PyTorch Implementation**: See lines 20-30 in `code/examples/single_gpu_training.py`

### 2. Fully Sharded Data Parallelism (FSDP)

**Concept**: Split model weights across GPUs, not just data.

**Key Points**:

- Model weights are distributed across GPUs
- Weights are broadcasted when needed
- Memory efficient for large models
- Communication: 3x model size per forward/backward pass

**Memory Savings**:

- 1B parameters = 8 GB memory
- With 80 GPUs: 10 GB per GPU

### 3. Hybrid Sharded Data Parallelism (HSDP)

**Concept**: Two-dimensional parallelism combining FSDP and DP.

**Key Points**:

- Groups of GPUs do FSDP internally
- Groups do DP with each other
- Optimizes for cluster topology
- Different communication patterns for different levels

### 4. Context Parallelism

**Concept**: Split sequences across GPUs for very long contexts.

**Key Points**:

- Parallelize over sequence dimension
- Challenging for attention computation
- Used for sequences >10,000 tokens
- Llama3 uses 16-way context parallelism for 130K sequences

**Attention Parallelization**:

- **Ring Attention**: Block-wise attention computation
- **Ulysses Attention**: Head-wise parallelism

### 5. Pipeline Parallelism

**Concept**: Split model layers across GPUs.

**Key Points**:

- Sequential dependencies between layers
- Use microbatches to reduce idle time
- "Bubble" time when GPUs are waiting
- Theoretical max efficiency: 1/N (N = number of pipeline stages)

**Optimization**:

- Multiple microbatches in flight
- Overlap communication and computation
- Llama3 uses 16-way pipeline parallelism

### 6. Tensor Parallelism

**Concept**: Split individual weight matrices across GPUs.

**Key Points**:

- Split matrices into blocks
- Special trick for consecutive layers
- Works well with transformer MLPs
- Llama3 uses 8-way tensor parallelism

## Memory Optimization Techniques

### Activation Checkpointing

**Problem**: Large models run out of memory storing activations.

**Solution**: Recompute activations during backward pass.

**Trade-offs**:

- **Memory**: O(√N) instead of O(N)
- **Compute**: O(N√N) instead of O(N)
- **Checkpoint Frequency**: Every √N layers

**Implementation**: See lines 15-25 in `code/examples/activation_checkpointing.py`

## Performance Optimization

### Model Flops Utilization (MFU)

**Definition**: Fraction of theoretical GPU throughput used for model computation.

**Formula**:

**Formula**: See lines 235-237 in `code/examples/mfu_benchmarking.py`

**Benchmarking**: See lines 50-70 in `code/examples/mfu_benchmarking.py`

**Target MFU**:

- **Good**: >30%
- **Excellent**: >40%
- **State-of-the-art**: 35-45%

### Scaling Recipe

1. **<1B parameters, <128 GPUs**: Use Data Parallelism
2. **1B-10B parameters**: Switch to FSDP + Activation Checkpointing
3. **>256 GPUs**: Use HSDP
4. **>1K GPUs, >50B parameters**: Add Context/Pipeline/Tensor Parallelism

## GPU Hardware Deep Dive

### NVIDIA H100 Architecture

The H100 represents the current state-of-the-art in GPU hardware for deep learning:

#### Memory Hierarchy

- **HBM Memory**: 80 GB at 3 TB/s bandwidth
- **L2 Cache**: 50 MB (much faster access)
- **L1 Cache**: 256 KB per SM (fastest access)

#### Compute Cores

- **132 Streaming Multiprocessors (SMs)**
- **128 FP32 cores per SM**: 256 FLOPS per SM per clock cycle
- **4 Tensor cores per SM**: 4,096 FLOPS per SM per clock cycle

#### Key Features

- **Mixed Precision**: 16-bit inputs, 32-bit accumulations
- **Tensor Cores**: Specialized for matrix multiplication
- **Memory Bandwidth**: 3 TB/s between compute and memory

### Performance Evolution

| GPU  | Year | FP32 TFLOPs | Tensor TFLOPs | Memory |
| ---- | ---- | ----------- | ------------- | ------ |
| K40  | 2013 | 5           | 0             | 12 GB  |
| P100 | 2016 | 10          | 0             | 16 GB  |
| V100 | 2017 | 15          | 125           | 32 GB  |
| A100 | 2020 | 20          | 312           | 80 GB  |
| H100 | 2022 | 67          | 989           | 80 GB  |
| B200 | 2025 | 83          | 5,000         | 192 GB |

## GPU Clusters and Infrastructure

### Llama3-405B Training Cluster

Meta's training infrastructure for Llama3-405B:

#### Cluster Architecture

- **Total GPUs**: 24,576 H100s
- **Memory**: 1.8 TB HBM memory
- **Compute Cores**: 415M FP32 cores, 13M tensor cores
- **Peak Performance**: 24 exaflops/second

#### Hierarchy

1. **Single GPU**: 80 GB HBM, 3 TB/s bandwidth
2. **GPU Server**: 8 GPUs, 900 GB/s inter-GPU communication
3. **Server Rack**: 16 GPUs (2 servers), 6 feet tall
4. **GPU Pod**: 3,072 GPUs (192 racks), 50 GB/s inter-rack
5. **Full Cluster**: 24,576 GPUs (8 pods)

### Communication Bandwidth

| Level           | Bandwidth | Relative Speed |
| --------------- | --------- | -------------- |
| GPU Internal    | 3 TB/s    | 1x             |
| Server Internal | 900 GB/s  | 0.3x           |
| Pod Internal    | 50 GB/s   | 0.017x         |
| Cluster Wide    | <50 GB/s  | <0.017x        |

## Resources

### Research Papers

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Ring Attention with Blockwise Parallelism for Long Sequences](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [FairScale: A General Purpose Modular PyTorch Library for High Performance and Large Scale Training](https://github.com/facebookresearch/fairscale)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)
- [Horovod: Fast and Easy Distributed Deep Learning in TensorFlow](https://arxiv.org/abs/1802.05799)
- [The Computational Limits of Deep Learning](https://arxiv.org/abs/2007.05558)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### Documentation

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [NVIDIA NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)

### Tools and Frameworks

- **PyTorch**: Primary framework for distributed training
- **DeepSpeed**: Microsoft's optimization library
- **FairScale**: Meta's scaling library
- **Megatron-LM**: NVIDIA's large-scale training framework

### Hardware

- **NVIDIA H100**: Current state-of-the-art GPU
- **Google TPU v5p**: Alternative training hardware
- **AWS Trainium**: Amazon's custom training chip

### Course Content

- **CS231N Lecture 11**: Large Scale Distributed Training (Spring 2025) - [Video Link](https://www.youtube.com/watch?v=9MvD-XsowsE)

## Code Examples

> **Note**: All examples are optimized for single GPU setups (T4/P100/V100).

### Single GPU Training

**Implementation**: See lines 1-50 in `code/examples/single_gpu_training.py`

### MFU Benchmarking (Performance Analysis)

**File**: `code/examples/mfu_benchmarking.py` (lines 1-402)

Key features:

- Model Flops Utilization calculation
- Performance benchmarking across different configurations
- Memory vs speed trade-off analysis

### Activation Checkpointing (Memory Optimization)

**File**: `code/examples/activation_checkpointing.py` (lines 1-433)

Key features:

- Memory vs compute trade-off analysis
- Adaptive checkpointing strategies
- Performance benchmarking with different checkpoint frequencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for GPU hardware and software tools
- Meta AI for open-source models and training details
- Microsoft Research for DeepSpeed and optimization techniques
- Google Research for PaLM and scaling research
- The PyTorch team for the distributed training framework

---

**Note**: This repository draws from multiple sources including academic papers, industry research, and educational materials. The content is for educational purposes and represents the state of distributed training as of 2025.
