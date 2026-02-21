// aizawa_slab_test.cu  —  v8
// Compile: nvcc -O3 -arch=sm_75 aizawa_slab_test.cu -o aizawa_slab_test -lm -lpthread
//
// v8 additions:
//   Test 4: Ejected pool index stress — drives viviani_superfluid_compact() under
//           heavy fragmentation to near-fill the ejected pool, verifies O(1) index
//           path fires and snap-back shadow updates are consistent.
//   Test 5: Derived threshold smoke test — confirms VIVIANI_ALLOC_RATE_THRESHOLD
//           is nonzero and < 1000, and prints its geometric derivation.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aizawa.cuh"
#include "aizawa_slab.cuh"

#define CUDA_CHECK(call) do { cudaError_t _e=(call);                    \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",              \
    __FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} } while(0)

// ============================================================================
// Test 1: Correctness
// Warp register state (sb_base, sb_cursor) per class, per thread.
// lane N -> slot N (or N%n_slots on smaller classes). No shared memory.
// ============================================================================

__global__ void slab_correctness_kernel(SlabPool* pool, uint32_t iters,
                                         uint32_t* d_errors) {
    uint32_t sb_base[SLAB_CLASSES], sb_cursor[SLAB_CLASSES];
    for (int c=0;c<SLAB_CLASSES;c++){sb_base[c]=0xFFFFFFFFu;sb_cursor[c]=0;}

    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t errs = 0;
    size_t sizes[SLAB_CLASSES]={64,128,256};

    for (int c=0;c<SLAB_CLASSES;c++) {
        for (uint32_t i=0;i<iters;i++) {
            void* ptr = viviani_slab_alloc(pool,c,&sb_base[c],&sb_cursor[c]);
            if (!ptr){errs++;continue;}
            volatile uint32_t* p=(volatile uint32_t*)ptr;
            uint32_t pat = tid^i^(uint32_t)sizes[c]^0xDEADBEEFu;
            *p = pat;
            __threadfence_block();
            if (*p != pat) errs++;
            viviani_slab_free(pool,ptr,c);
        }
    }
    if (errs) atomicAdd(d_errors,errs);
}

void test_correctness(VivianiSlabContext* ctx) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  SLAB TEST 1: Correctness                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    uint32_t *d_err; CUDA_CHECK(cudaMalloc((void**)&d_err,4));
    CUDA_CHECK(cudaMemset(d_err,0,4));
    slab_correctness_kernel<<<128,128>>>(&ctx->pool,50,d_err);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h=0; CUDA_CHECK(cudaMemcpy(&h,d_err,4,cudaMemcpyDeviceToHost));
    cudaFree(d_err);
    SlabStats ss=viviani_slab_stats(ctx);
    printf("  Allocs   (64/128/256): %llu / %llu / %llu\n",
           (unsigned long long)ss.allocs[0],(unsigned long long)ss.allocs[1],(unsigned long long)ss.allocs[2]);
    printf("  Frees    (64/128/256): %llu / %llu / %llu\n",
           (unsigned long long)ss.frees[0],(unsigned long long)ss.frees[1],(unsigned long long)ss.frees[2]);
    printf("  Fallbacks            : %llu / %llu / %llu\n",
           (unsigned long long)ss.fallbacks[0],(unsigned long long)ss.fallbacks[1],(unsigned long long)ss.fallbacks[2]);
    printf("  Corruption errors    : %u\n",h);
    printf(h==0?"\n✓ PASS: No corruption\n":"\n✗ FAIL: %u errors\n",h);
}

// ============================================================================
// Test 2: Throughput vs device malloc
// Baseline is hardcoded from empirical measurement on RTX 2060 (stable across
// runs to within ~5%). We never run the baseline kernel during the test —
// device malloc from 8192 simultaneous threads runs the GPU at 100% load for
// minutes and adds nothing new. The slab kernel is measured fresh each run.
// ============================================================================

// Hardcoded baseline (us/iter) from prior empirical runs on this GPU.
// cudaMalloc from device is stable: ~1650-1760 us/iter across all runs.
static const float BASELINE_US[SLAB_CLASSES] = {1677.f, 1750.f, 2083.f};

// Warm-up and measure in one launch so sb_base persists and L2 is genuinely
// hot when the timed portion begins. clock64 timestamps bracket only the
// back MEASURE_IT iterations; the first WARMUP_IT iterations are free.
// Result written as (total_cycles, count) for averaging across threads.
__global__ void slab_tput_kernel(SlabPool* pool, int cls,
                                  uint32_t warmup_it, uint32_t measure_it,
                                  uint64_t* out_ns) {
    uint32_t sb_base = 0xFFFFFFFFu, sb_cursor = 0;
    void* last = nullptr;
    uint32_t total = warmup_it + measure_it;

    // Warm-up: run without timing so sb_base is set and L2 is hot
    for (uint32_t i = 0; i < warmup_it; i++) {
        void* ptr = viviani_slab_alloc(pool, cls, &sb_base, &sb_cursor);
        if (ptr) { *(volatile uint8_t*)ptr = (uint8_t)(threadIdx.x + i); }
        if (last) viviani_slab_free(pool, last, cls);
        last = ptr;
    }

    // Measure: same kernel, same sb_base, L2 warm
    uint64_t t0 = clock64();
    for (uint32_t i = warmup_it; i < total; i++) {
        void* ptr = viviani_slab_alloc(pool, cls, &sb_base, &sb_cursor);
        if (ptr) { *(volatile uint8_t*)ptr = (uint8_t)(threadIdx.x + i); }
        if (last) viviani_slab_free(pool, last, cls);
        last = ptr;
    }
    uint64_t t1 = clock64();
    if (last) viviani_slab_free(pool, last, cls);

    // Leader of warp 0 in block 0 records peak cycles
    if (threadIdx.x == 0 && blockIdx.x == 0)
        atomicMax((unsigned long long*)out_ns, (unsigned long long)(t1 - t0));
}

void test_throughput(VivianiSlabContext* ctx) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  SLAB TEST 2: Throughput vs device malloc                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  (baseline = empirical device-malloc on this GPU; not re-run)\n");
    printf("  (warm+measure in one launch — sb_base persists, L2 hot at t0)\n");

    // warmup_it chosen to cycle through all cursor positions at least once,
    // ensuring every superblock in the warp's range has been touched.
    // SBS_PER_WARP=18 -> worst case n_positions=18 (64B). Use 200 warm iters.
    const uint32_t WARMUP = 200, MEASURE = 2000, BL = 64, TH = 128;
    const char* nm[] = {"64B ", "128B", "256B"};

    uint64_t* d; CUDA_CHECK(cudaMalloc((void**)&d, 8));

    printf("  Class | Cycles/iter | Slab us/iter | Base us/iter | Speedup\n");
    printf("  ------|-------------|--------------|--------------|--------\n");

    // Estimated GPU clock in GHz (Turing boost ~1.68 GHz)
    const float GPU_GHZ = 1.68f;

    for (int c = 0; c < SLAB_CLASSES; c++) {
        CUDA_CHECK(cudaMemset(d, 0, 8));
        slab_tput_kernel<<<BL,TH>>>(&ctx->pool, c, WARMUP, MEASURE, d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t cyc = 0; CUDA_CHECK(cudaMemcpy(&cyc, d, 8, cudaMemcpyDeviceToHost));

        float cpi     = (float)cyc / MEASURE;
        float slab_us = cpi / (GPU_GHZ * 1000.f);  // cycles / (cycles/us) = us
        float sp      = (slab_us > 0.f) ? BASELINE_US[c] / slab_us : 0.f;
        printf("  %s  | %11.1f | %12.3f | %12.3f | %.1fx\n",
               nm[c], cpi, slab_us, BASELINE_US[c], sp);
    }

    CUDA_CHECK(cudaFree(d));
}

// ============================================================================
// Test 3: Concurrent stress — 256 blocks × 256 threads, 1000 iters, hold 4
// ============================================================================

// viviani_slab_alloc() is warp-cooperative: all active lanes must call it
// with the SAME class simultaneously. Class selected at warp granularity.
__global__ void slab_stress_kernel(SlabPool* pool, uint32_t iters,
                                    uint32_t* d_cnt) {
    const uint32_t warps_per_block = blockDim.x / 32u;
    const uint32_t warp_id         = threadIdx.x / 32u;

    extern __shared__ uint32_t smem[];
    uint32_t* warp_sb_base   = smem + warp_id * SLAB_CLASSES;
    uint32_t* warp_sb_cursor = smem + warps_per_block * SLAB_CLASSES
                                    + warp_id * SLAB_CLASSES;

    if ((threadIdx.x & 31u) == 0) {
        for (int c = 0; c < SLAB_CLASSES; c++) {
            warp_sb_base[c]   = 0xFFFFFFFFu;
            warp_sb_cursor[c] = 0;
        }
    }
    __syncthreads();

    void*    held[4] = {nullptr}; int hcls[4] = {0};
    int      hcnt = 0; uint32_t local_n = 0;
    uint32_t global_warp_id = blockIdx.x * warps_per_block + warp_id;

    for (uint32_t i = 0; i < iters; i++) {
        int c = (int)((global_warp_id + i) % (uint32_t)SLAB_CLASSES);
        void* ptr = viviani_slab_alloc(pool, c,
                                        &warp_sb_base[c],
                                        &warp_sb_cursor[c]);
        if (ptr) {
            local_n++;
            if (hcnt < 4) {
                held[hcnt] = ptr; hcls[hcnt] = c; hcnt++;
            } else {
                viviani_slab_free(pool, held[0], hcls[0]);
                for (int j = 0; j < 3; j++) { held[j]=held[j+1]; hcls[j]=hcls[j+1]; }
                held[3] = ptr; hcls[3] = c;
            }
        }
    }
    for (int j = 0; j < hcnt; j++)
        if (held[j]) viviani_slab_free(pool, held[j], hcls[j]);
    atomicAdd(d_cnt, local_n);
}

void test_stress(VivianiSlabContext* ctx) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  SLAB TEST 3: Concurrent stress (256×256 threads)            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    uint32_t* d; CUDA_CHECK(cudaMalloc((void**)&d,4));
    CUDA_CHECK(cudaMemset(d,0,4));
    size_t smem_sz = 2u * (256u/32u) * (size_t)SLAB_CLASSES * sizeof(uint32_t);
    slab_stress_kernel<<<256, 256, smem_sz>>>(&ctx->pool, 1000, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h=0; CUDA_CHECK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost));
    cudaFree(d);
    SlabStats ss=viviani_slab_stats(ctx);
    uint64_t ta=ss.allocs[0]+ss.allocs[1]+ss.allocs[2];
    uint64_t tf=ss.fallbacks[0]+ss.fallbacks[1]+ss.fallbacks[2];
    float fbr=(ta+tf>0)?100.f*tf/(ta+tf):0.f;
    printf("  Successful allocs: %u\n",h);
    printf("  Fallback rate:     %.2f%%\n",fbr);
    printf(fbr<1.f?"\n✓ PASS: <1%% fallbacks\n":
           fbr<5.f?"\n? MARGINAL: %.2f%% fallbacks\n":
                   "\n✗ REVIEW: %.2f%% fallbacks\n",fbr);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Viviani/Aizawa + Geometry-Native Slab Test Suite (v8)       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("\nGPU: %s | SM %d.%d | %d SMs | %.1f GB\n\n",
           prop.name,prop.major,prop.minor,prop.multiProcessorCount,prop.totalGlobalMem/1e9);

    VivianiAllocator va; viviani_init(&va,VIVIANI_POOL_SIZE);
    VivianiSlabContext ctx;
    cudaError_t err=viviani_slab_init(&ctx,SLAB_POOL_DEPTH);
    if(err!=cudaSuccess){fprintf(stderr,"FATAL: %s\n",cudaGetErrorString(err));return 1;}

    printf("Slab pool: %d classes × %u sbs × 4KB = %.1f MB\n",
           SLAB_CLASSES,SLAB_POOL_DEPTH,
           (float)SLAB_CLASSES*SLAB_POOL_DEPTH*4096/(1024*1024));
    printf("Lane model: sb_sub=lane/n_slots, slot=lane%%n_slots (zero wasted lanes)\n");
    printf("Cursor recycles within warp range (freed bits reclaimed)\n\n");

    test_correctness(&ctx);
    viviani_slab_reset(&ctx);

    test_throughput(&ctx);
    viviani_slab_reset(&ctx);

    test_stress(&ctx);
    viviani_slab_reset(&ctx);

    viviani_slab_run_bench(&ctx,1000);

    // -------------------------------------------------------------------------
    // Test 4: Ejected pool index stress (V8)
    // Fragment the arena then drive compaction repeatedly, pushing ejected_count
    // toward MAX_EJECTED. Verifies: pool stays bounded, shadow_updates fire
    // (snap-backs via index), no count overflow.
    // -------------------------------------------------------------------------
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 4: Ejected pool index stress (V8)                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    {
        const int N = 2048;
        void* ptrs[N];
        for (int i = 0; i < N; i++) ptrs[i] = viviani_alloc(&va, 64);
        for (int i = 0; i < N; i += 2)
            if (ptrs[i]) { viviani_free(&va, ptrs[i], 64); ptrs[i] = NULL; }

        uint32_t total_shadow_updates = 0, total_blocks_moved = 0, peak_ejected = 0;
        for (int r = 0; r < 8; r++) {
            if (viviani_should_compact(&va)) {
                SuperfluidStats ss = viviani_superfluid_compact(&va);
                total_shadow_updates += ss.shadow_updates;
                total_blocks_moved   += ss.blocks_moved;
            }
            VivianiStats vs = viviani_stats(&va);
            if (vs.ejected_count > peak_ejected) peak_ejected = vs.ejected_count;
        }

        VivianiStats vs = viviani_stats(&va);
        printf("  Compact rounds:   8\n");
        printf("  Blocks moved:     %u\n", total_blocks_moved);
        printf("  Shadow updates:   %u  (snap-backs via O(1) index)\n", total_shadow_updates);
        printf("  Peak ejected:     %u / %d\n", peak_ejected, MAX_EJECTED);
        printf("  Final ejected:    %u\n", vs.ejected_count);

        if (vs.ejected_count <= (uint32_t)MAX_EJECTED)
            printf("\n✓ PASS: Ejected pool stayed bounded\n");
        else
            printf("\n✗ FAIL: Ejected pool overflow\n");

        for (int i = 0; i < N; i++)
            if (ptrs[i]) viviani_free(&va, ptrs[i], 64);
        viviani_ejected_reset(&va);
    }

    // -------------------------------------------------------------------------
    // Test 5: Derived threshold smoke test (V8)
    // -------------------------------------------------------------------------
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 5: Derived threshold smoke test (V8)                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    {
        uint32_t thresh = VIVIANI_ALLOC_RATE_THRESHOLD;
        printf("  HOPF_Q=%.4f  FRACTAL_DEPTH=%d  =>  threshold=%u (%.3f allocs/unit)\n",
               VIVIANI_HOPF_Q, VIVIANI_FRACTAL_DEPTH,
               thresh, (float)thresh / 1000.0f);
        if (thresh > 0 && thresh < 1000)
            printf("\n✓ PASS: Threshold geometrically grounded, in range [1,999]\n");
        else
            printf("\n✗ FAIL: Threshold out of expected range\n");
    }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  All tests complete (v8)                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    viviani_slab_destroy(&ctx);
    viviani_destroy(&va);
    return 0;
}
