

# **Triton Fused Attention Kernel (CUDA + ROCm)**

Modern large language models rely heavily on **scaled dot-product attention**, an operation that dominates both compute time and memory traffic during inference. Improving this single component can meaningfully shift the end-to-end performance profile of an entire model, especially under high-throughput serving conditions.

This project explores the design, implementation, and cross-platform behavior of a **fused attention kernel written in Triton**, targeting both **NVIDIA CUDA** and **AMD ROCm** backends. It is structured as an experiment in bringing high-performance attention kernels to multiple GPU architectures, understanding backend differences, and building the kind of infrastructure needed for scalable multi-platform LLM inference.

---

# **Why This Project Exists**

As AI workloads move toward heterogeneous GPU clusters, inference frameworks must run reliably across different accelerator vendors.
This requires:

* kernels that compile on multiple backends
* consistent numerical behavior across architectures
* stable performance under different memory and execution models
* the ability to reason about codegen and execution at a low level

Triton offers a Pythonic approach to kernel authoring that compiles into highly optimized GPU code via MLIR and LLVM. But CUDA and ROCm differ in subtle ways—register pressure, wavefront/warp differences, memory hierarchies, and vectorization patterns.

This project uses a **single fused attention kernel** as a microcosm of real inference work:

* fused QKᵀ calculation
* stable softmax with streaming normalization
* multiplication by V
* optional causal masking
* batched multi-head execution
* CUDA vs ROCm backend behavior

This captures the core performance-critical path of attention while illuminating cross-platform differences.

---

# **Project Goals**

### **1. Implement a realistic fused attention kernel in Triton**

The kernel is decomposed into:

* QKᵀ dot-product in tiles
* streaming max/sum accumulation (stable softmax)
* causal mask support
* softmax normalization
* weighted value aggregation (probs @ V)
* multi-head, batched wrapper

Although simplified relative to FlashAttention v2, it uses the same foundational ideas:
tiling, on-chip accumulation, reduced memory accesses, and kernel fusion.

---

### **2. Run the same kernel across CUDA + ROCm**

The project validates backend consistency:

* does the kernel compile cleanly on both backends?
* do outputs match PyTorch attention?
* do performance characteristics differ between CUDA and ROCm?
* where does the backend-specific codegen diverge?

This is a real-world challenge when enabling inference on new GPU platforms.

---

### **3. Benchmark and profile**

The repository includes tools to compare:

* PyTorch baseline scaled dot-product attention
* Triton fused kernel (CUDA backend)
* Triton fused kernel (ROCm backend)

Metrics of interest:

* runtime per token
* prefill throughput
* FLOP utilization
* memory bandwidth usage
* occupancy metrics
* wavefront/warp behavior
* register pressure

Profiling utilities (Nsight + rocprof) are scaffolded in the repo.

---

### **4. Document architectural differences**

Key questions explored:

* How does Triton tile differently on AMD vs NVIDIA?
* Do different vector widths or wave sizes affect throughput?
* Is the bottleneck compute bound or memory bound?
* Does register pressure push ROCm kernels into spills?
* Where does MLIR/LLVM codegen differ between the two backends?

This builds intuition needed for multi-platform inference engineering.

---

### **5. Build a structure that resembles real inference kernel development**

The repository is intentionally structured like a kernel developer’s workspace:

```
kernels/       – Triton fused attention kernel(s)
tests/         – correctness tests (vs PyTorch attention)
benchmarks/    – timing + throughput benchmarks
analysis/      – profiling notes and architectural findings
scripts/       – run/profile helpers for CUDA + ROCm
docs/          – design decisions, tuning notes
```

This mirrors how GPU engineers structure performant kernels for LLM inference systems like vLLM, FasterTransformer, or custom internal runtimes.

---

# **Fused Attention in Triton**

The kernel is built around three tiled loops:

### **Tile 1 — QKᵀ (Attention Scores)**

* Reads blocks of Q and K
* Computes blockwise dot-products
* Applies scaling and optional causal masking
* Updates blockwise softmax stats (max, sum)

### **Tile 2 — Softmax (Stabilization)**

* Uses the accumulated max/sum
* Applies exponentiation and normalization

### **Tile 3 — Multiply by V**

* Reads V tiles
* Aggregates weighted value vectors
* Writes output back to global memory

The structure avoids materializing the full L×L attention matrix, reduces DRAM reads/writes, and keeps intermediate values on-chip as much as possible.

---

# **Correctness Testing**

Under `tests/`, we:

* run PyTorch’s reference attention implementation
* compare outputs to the Triton kernel
* test both causal and non-causal modes
* use configurable tolerances
* test multiple shapes and data patterns

Correctness is validated for FP32 initially, with potential extensions to BF16/FP16.

---

# **Benchmarking**

Benchmarks under `benchmarks/` measure:

* Triton kernel latency
* PyTorch baseline latency
* CUDA vs ROCm backend performance
* Effect of varying (BLOCK_M, BLOCK_N, BLOCK_D) tile sizes

Performance insights and example numbers are recorded in:

```
benchmarks/results.md
docs/findings.md
```

---

# **Profiling**

The project includes:

* `scripts/profile_cuda.sh` (Nsight Systems / Nsight Compute)
* `scripts/profile_rocm.sh` (rocprof)

Profiling goals include:

* kernel timeline
* memory load/store efficiency
* warp/wavefront utilization
* instruction mix
* divergence patterns
* spills/stalls
* bandwidth ceilings

The file `analysis/profiling_notes.md` records insights.

---

# **Cross-Platform Observations**

As development progresses, key insights are logged:

* Where CUDA outperforms ROCm
* Where ROCm outperforms CUDA
* Backend-specific optimizations
* Differences in kernel launch configs
* Impact of wavefront size (64) vs warp size (32)
* Differences in vectorization and memory coalescing
* How Triton’s MLIR pipeline behaves differently

These are exactly the kinds of insights needed to optimize inference kernels for new hardware.

---

# **Future Extensions**

Planned enhancements include:

* FlashAttention-style block-sparse attention
* Masked and blockwise attention variants
* Mixed precision (FP16/BF16) kernels
* Integration into a mini-inference engine
* Auto-tuner for tile sizes on CUDA and ROCm
* Multi-GPU experiments via RCCL/NCCL

---

# **Summary**

This repository is a self-contained exploration of:

* writing fused LLM kernels in Triton
* making them work across NVIDIA and AMD GPUs
* validating correctness and performance
* building tooling around kernels (tests, benchmarks, profiling)
* developing intuition for cross-platform inference performance

The focus is not just kernel code—it’s **kernel engineering**, **performance debugging**, and **multi-platform inference enablement**, which mirrors real-world work done by GPU kernel engineers and inference teams at frontier AI labs.
