Patent Pending. Free for -personal-, individual research use.  
For commercial use please inquire.

# Viviani Hopfion Allocator — Aizawa Quark-Level Superfluid Stirring

**A self-healing, topologically-protected, geometry-native memory allocator** inspired by microtubule dynamics, Hopfions, and the Aizawa strange attractor.

Built from first principles with one core philosophy:  
**"Everything correct stays untouched forever."**

## Overview

This allocator unifies CPU and GPU allocation under a single coherent physical model:

- **CPU side**: Forward-bias bump allocator with Viviani curve offsets (5D recirculation), Hopf invariants for topological protection, geometric repair, viral self-healing, and Aizawa quark stirring for defect ejection/snap-back.
- **GPU side**: Geometry-native slab allocator (V8) — warp-permanent ownership ranges, zero-waste lane model, Viviani scatter for contention-free refills, and full warp-uniform control flow.

The result is an allocator that behaves like microtubule/DNA physics: defects self-organize, get ejected into chaos, heal via Hopf braiding, and return to coherent flow — all while delivering **hundreds to thousands of times faster** small-allocation performance than `cudaMalloc`.

---

## Features

### Core Design
- **Viviani 5D Recirculation** — allocations follow a true laziness manifold instead of linear addresses.
- **Hopf Invariant Protection** — topological charge prevents corruption propagation.
- **Aizawa Quark Stirring** — stressed blocks eject into the strange attractor and snap back with conserved winding.
- **Superfluid Compaction** — self-healing defragmentation that recovers space while preserving invariants.
- **Geometric Repair** — DNA-style mismatch detection and viral propagation of fixes.

### GPU Slab Allocator (V8)
- Warp-permanent ownership ranges (no global atomics after initial claim)
- Zero-waste lane model (`sb_sub = lane / n_slots`, `slot = lane % n_slots`)
- Viviani scatter prevents hot-spot contention
- Full warp-uniform control flow (no divergence)
- SLAB_ATOMIC_* portability macros for future AMD/HIP targeting
- 507×–850× speedup over `cudaMalloc` for 64/128/256 B allocations on RTX 2060

### Test Suite Results (RTX 2060, SM 7.5)
- **Correctness**: 0 corruption errors across 819,200 allocs/frees per size class
- **Stress (256×256 threads)**: 65+ million successful allocs, **0.00% fallback rate**
- **Throughput**: 507× (64B) / 560× (128B) / 850× (256B) vs device malloc
- **Occupancy**: 100% theoretical across all kernels

---

## Building & Running

### Allocator — Prerequisites

The core allocator (`aizawa.cuh`, `aizawa_slab.cuh`) and test suite have no dependencies beyond CUDA itself.

```bash
# Arch Linux
sudo pacman -S cuda

# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Fedora
sudo dnf install cuda
```

### Allocator — Compile & Test

```bash
nvcc -O3 -arch=sm_75 -std=c++17 aizawa_slab_test.cu -o slab_test
./slab_test
```

Adjust `-arch=` to match your GPU: `sm_61` = GTX 10xx, `sm_75` = RTX 20xx, `sm_86` = RTX 30xx, `sm_89` = RTX 40xx.

---

## Blackhole Demo — Hopfion Accretion Disk

A realtime physics simulation of a Schwarzschild black hole accretion disk. Each particle runs full Keplerian orbital mechanics, Aizawa quark ejection dynamics near the ISCO, Doppler beaming, and gravitational redshift. The background is a raymarched Schwarzschild metric with photon ring and gravitational lensing.

All particles are allocated and tracked through the Viviani allocator, making this a live stress test of the allocator under continuous churn.

### Physics Model
- Keplerian orbital mechanics (ω ∝ r⁻³/²)
- Shakura-Sunyaev thin disk temperature profile (T ∝ r⁻³/⁴)
- Doppler beaming — blueshift on approaching side, redshift on receding
- Gravitational redshift: √(1 − rₛ/r)
- Aizawa ejection near ISCO models magneto-rotational instability
- Viviani groove density wave in the fragment shader

### Blackhole Demo — Prerequisites

Requires CUDA plus OpenGL windowing libraries.

```bash
# Arch Linux
sudo pacman -S cuda glfw-wayland glew

# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit libglfw3-dev libglew-dev

# Fedora
sudo dnf install cuda glfw-devel glew-devel
```

### Blackhole Demo — Compile

```bash
nvcc -O3 -arch=sm_75 -std=c++17 blackhole.cu -lglfw -lGLEW -lGL -o blackhole
```

### Blackhole Demo — Run

```bash
./blackhole                    # Hex lattice, 80 rings (~19k points)
./blackhole --rings 150        # Denser hex lattice (~68k points)
./blackhole --random           # 20k random points, physics-driven evolution
./blackhole --random 500000    # 500k random points (heavy)
./blackhole --help             # Full usage
```

### Controls
| Input | Action |
|-------|--------|
| Left drag | Orbit camera |
| Scroll | Zoom |
| R | Reset camera + allocator ejected pool |
| Space | Pause / resume simulation |
| ESC | Quit |

---

## Project Structure

```
aizawa.cuh            — CPU allocator (Viviani, Hopf, Aizawa, compaction)
aizawa_slab.cuh       — GPU slab allocator (V8, warp-native)
aizawa_slab_test.cu   — Allocator test suite
blackhole.cu          — Accretion disk simulation / allocator stress test
```
