# Metal Backend Performance Optimization Roadmap

> **Goal:** Achieve at least **3× total speed-up** over the current CPU implementation of GrabCut and other Metal-accelerated algorithms, while maintaining output parity with the reference OpenCV CPU backend.

---

## 📈 Current Baseline

| Stage | CPU (ms) | Metal (ms) | Speed-up |
|-------|----------|------------|----------|
| Initialization (0 iters) | 238 ms | 195 ms | **1.22×** |
| 1 Iteration | 3 528 ms | 3 020 ms | **1.17×** |
| 3 Iterations | 7 515 ms | 5 288 ms | **1.42×** |
| 5 Iterations | 10 027 ms | 7 748 ms | **1.29×** |

**Primary Bottleneck:** CPU graph-cut (≈ 60 % of iteration time)

---

## 🛣️ Roadmap Overview

| Phase | Target Speed-up | Key Deliverables | ETA |
|-------|-----------------|------------------|-----|
| **P0** | 1.5×-2× | • Eliminate per-iteration `stream.syncCPU()` <br>• Consolidate downloads <br>• Persistent Metal buffers | **2 weeks** |
| **P1** | 2×-3× | • Full GPU graph-cut (push-relabel or Boykov-Kolmogorov) <br>• In-GPU unary/pairwise construction | **6 weeks** |
| **P2** | 3×-4× | • Overlap compute & transfers (command-buffer chaining) <br>• Batched multi-image pipeline | **10 weeks** |
| **P3** | 4×-5× | • Algorithm-specific kernels (GrabCut, Morphology, Connected-Components) <br>• MetalFX / tile-render fallback for >16 MP images | **14 weeks** |

---

## 📍 Detailed Action Items

### Phase 0 — Low-Hanging Fruit (1.5×-2×)

| ID | Task | Owner | Notes |
|----|------|-------|-------|
| P0-1 | **Remove per-iteration `stream.syncCPU()` (clustering.mm L620)** | @alan | Replace with fenced buffer uploads; rely on implicit commit in `convertAndEncode()` |
| P0-2 | **Batch downloads** | @maria | Use `StreamAccessor::downloadMany()` helper to download FG/BG terms in one pass |
| P0-3 | **Persistent buffers** for `pairwiseWeights` & `components` | @wei | Allocate once in `GrabCutImpl` ctor, reuse across iterations |
| P0-4 | **Unit test** – verify no functional regression | QA | `EXPECT_MAT_NEAR` (tolerance ≤ 1e-5) |

### Phase 1 — Full GPU Graph-Cut (2×-3×)

| ID | Task | Owner | Notes |
|----|------|-------|-------|
| P1-1 | **Design MetalGraphCut API** (`MetalGraphCut.hpp`) | @alex | Expose `buildGraph()` / `solve()` / `getSegmentation()` |
| P1-2 | **Implement parallel push-relabel** kernel set | @liu | Use shared memory for excess & height; 16×16 tiles |
| P1-3 | **Integrate into GrabCutImpl** (guard with `USE_METAL_GRAPHCUT`) | @alan | Remove CPU GCGraph fallback path |
| P1-4 | **Performance regression tests** | QA | Target >2× speed-up vs P0 build |

### Phase 2 — Pipelining & Streaming (3×-4×)

| ID | Task | Owner | Notes |
|----|------|-------|-------|
| P2-1 | **Command-buffer chaining** for iterative kernels | @maria | Reuse same command-buffer object across iterations |
| P2-2 | **Asynchronous transfer staging buffer** | @wei | Use `blitEncoder.copyFromTexture` → `MTLBuffer` without CPU sync |
| P2-3 | **Batched image processing API** (`processBatch(std::vector<MetalMat>)`) | @alex | Hide latency in multi-image workflows |
| P2-4 | **Scalability tests** – 0.3 MP → 16 MP | QA | Validate memory footprint < 512 MB |

### Phase 3 — Algorithm Specialization (4×-5×)

| ID | Task | Owner | Notes |
|----|------|-------|-------|
| P3-1 | **Optimized morphology kernels** (separable + subgroup) | @liu | Dilate/Erode 3× faster than CPU SSE 4.2 |
| P3-2 | **Connected-Components labeling (CCLabel)** | @alan | Parallel union-find with atomicOps; reuse from Vulkan port |
| P3-3 | **MetalFX tile-render fallback** | @maria | Process >16 MP images; merge tiles in GPU |
| P3-4 | **Energy-based GrabCut variant** | Research | Evaluate MPSGraph differentiable solver |

---

## ✅ Acceptance Criteria

1. **Speed:** End-to-end GrabCut (5 iters, 3 MP) ≤ 3 s (**>3× faster** than baseline CPU)
2. **Accuracy:** IoU with CPU reference ≥ 0.80 (no regression)
3. **Memory:** Peak GPU memory ≤ 1 GB for 4K images
4. **Stability:** 100× stress test with random seeds – no leaks, no crashes
5. **CI:** All new kernels covered by unit and perf tests (`--gtest_filter="*Metal*"`)

---

## 📊 Metrics Dashboard (to be automated)

| Metric | Baseline | Target |
|--------|----------|--------|
| Init Time (ms) | 195 | ≤ 100 |
| Per-Iter (ms) | 1 556 | ≤ 500 |
| GPU ↔ CPU Transfers (MB) | 11.4 × 5 | ≤ 11.4 × 1 |
| IoU Accuracy | 0.846 | ≥ 0.80 |

All metrics will be captured by `opencv_perf_metalimgproc` and uploaded to the CI dashboard.

---

## 🔄 Review & Tracking

- **Weekly performance stand-up (@Tue 17:00)**
- JIRA board: `METAL-PERF-###`
- Status labels: `pending`, `in_progress`, `blocked`, `done`

---

*Last updated: 2025-07-25* 