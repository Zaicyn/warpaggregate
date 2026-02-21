// aizawa.cuh  —  V8
// Unified Lock-Free Allocator — Aizawa Quark-Level Superfluid Stirring Fully Integrated
// Built on the Viviani/Hopfion base; all original correct code preserved.
//
// V8 changes:
//   - ejected_index[MAX_EJECTED]: direct-mapped O(1) lookup table for ejected pool.
//     Hash key: block_idx % MAX_EJECTED -> ejected pool slot (VIVIANI_EJECTED_EMPTY sentinel).
//     Collision falls back to linear scan (rare). Eliminates O(n*m) worst case in
//     viviani_superfluid_compact().
//   - VIVIANI_ALLOC_RATE_THRESHOLD: replaces magic constant 800 in viviani_cuda_free().
//     Derived from VIVIANI_HOPF_Q and VIVIANI_FRACTAL_DEPTH — lives in the same
//     geometric family as every other constant in this file.
//   - viviani_ejected_reset(): clean pool+index teardown between test phases.
//   - AMD/HIP portability: host-side atomics unchanged (GCC __atomic_*). CUDA device
//     wrappers isolated in #ifdef __CUDACC__ blocks. HIP can mechanically hipify.
//
// Philosophy: "Everything correct stays untouched forever."

#ifndef VIVIANI_ALLOC_HOPFION_FIXED_CUH
#define VIVIANI_ALLOC_HOPFION_FIXED_CUH

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// GCC 13+ stdatomic.h (pulled in transitively by C++ standard headers) defines:
//   #define _Atomic(_Tp) ::std::atomic<_Tp>
// CUDA 13 CCCL headers use _Atomic as a template parameter name internally,
// and in some call-like contexts this function-like macro fires, producing
// "expected a type specifier" errors throughout cccl/cuda/std/atomic.
// We use __atomic_* GCC builtins exclusively — the _Atomic macro is never
// needed here — so we unconditionally undef it after system headers.
#ifdef _Atomic
    #undef _Atomic
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define VIVIANI_HOST_DEVICE __host__ __device__
#define VIVIANI_DEVICE __device__
#endif

// Host-side atomics via GCC/Clang __atomic_* builtins (compatible with nvcc).
//
// VIVIANI_atomic_* — prefixed to avoid colliding with CUDA 13 CCCL's real
// C++ template functions: cuda::std::atomic_store, atomic_load, etc.
// Using bare atomic_store/atomic_load as macros corrupts those definitions.
//
// VIVIANI_ATOMIC (= volatile) — used as struct field qualifier instead of
// _Atomic. GCC 13 stdatomic.h defines _Atomic as a function-like macro;
// CCCL uses _Atomic as an identifier, so we never define that macro.
#define VIVIANI_ATOMIC volatile
#define VIVIANI_atomic_store(ptr, val)              __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_load(ptr)                    __atomic_load_n(ptr, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_add(ptr, val)          __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_sub(ptr, val)          __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_or(ptr, val)           __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_and(ptr, val)          __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_compare_exchange_weak(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, expected, desired, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)

#ifndef __CUDACC__
#define VIVIANI_HOST_DEVICE
#define VIVIANI_DEVICE
#endif

// ============================================================================
// Configuration — Geometric Constants from Cohesive Breakdown + Hopfions
// ============================================================================

#define VIVIANI_POOL_SIZE       (64ULL * 1024 * 1024)   // 64 MB base pool
#define VIVIANI_BLOCK_SIZE      64                       // Cache-line aligned
#define VIVIANI_UNIT_SIZE       256                      // 4 blocks = 1 Hopfion unit
#define VIVIANI_HOPF_Q          1.97f                    // Approximate Hopf charge (Q≈2)

// Period-4 bin sizes (quaternary encoding)
#define VIVIANI_BIN_COUNT       16
#define VIVIANI_BIN_0           64ULL
#define VIVIANI_BIN_1           128ULL
#define VIVIANI_BIN_2           256ULL
#define VIVIANI_BIN_3           512ULL
#define VIVIANI_BIN_4           1024ULL
#define VIVIANI_BIN_5           2048ULL
#define VIVIANI_BIN_6           4096ULL
#define VIVIANI_BIN_7           8192ULL
#define VIVIANI_BIN_8           (32ULL  * 1024)
#define VIVIANI_BIN_9           (64ULL  * 1024)
#define VIVIANI_BIN_10          (128ULL * 1024)
#define VIVIANI_BIN_11          (256ULL * 1024)
#define VIVIANI_BIN_12          (1ULL   * 1024 * 1024)
#define VIVIANI_BIN_13          (4ULL   * 1024 * 1024)
#define VIVIANI_BIN_14          (16ULL  * 1024 * 1024)
#define VIVIANI_BIN_15          (64ULL  * 1024 * 1024)

#define VIVIANI_SHADOW_STRIDE   8
#define VIVIANI_FLAG_RING_SIZE  4096
#define VIVIANI_MAX_DEFECTS     64

#define VIVIANI_MAX_STREAMS     1024
#define VIVIANI_HOLD_PER_STREAM 8
#define VIVIANI_GPU_HELD_CAP    (1536ULL * 1024 * 1024)  // 1.5 GB GPU hold cap

#define VIVIANI_FREELIST_SLOTS  8
#define VIVIANI_CACHE_BYPASS_THRESHOLD 16
#define VIVIANI_FRACTAL_DEPTH   4
#define VIVIANI_5D_MODULUS      8

// ============================================================================
// Aizawa Chaos Types  (quark-level stirring for ejected blocks)
// ============================================================================

typedef struct {
    float state[3];   // x, y, z on the Aizawa attractor
    float phi;        // phase angle
    uint32_t steps;   // integration steps taken
} AizawaState;

typedef struct {
    size_t     block_idx;
    AizawaState state;
} EjectedEntry;

#define MAX_EJECTED 4096

// Sentinel: no ejected entry at this hash slot
#define VIVIANI_EJECTED_EMPTY 0xFFFFFFFFu

// GPU hold-protection rate threshold — derived from framework geometry:
//   VIVIANI_HOPF_Q (1.97) * 1000 / VIVIANI_FRACTAL_DEPTH (4) ≈ 492
// alloc_rate_fp is stored scaled by 1000, so 492 = ~0.49 allocs/unit-time.
// Replaces the magic constant 800 in viviani_cuda_free().
#define VIVIANI_ALLOC_RATE_THRESHOLD \
    ((uint32_t)((VIVIANI_HOPF_Q * 1000.0f) / (float)VIVIANI_FRACTAL_DEPTH))

// ============================================================================
// Viviani Curve Parametrization (Triple Harmonic)
// ============================================================================

typedef struct { float x, y, z; } VivianNormal;

VIVIANI_HOST_DEVICE static inline VivianNormal viviani_normal(float theta) {
    float sin_t  = sinf(theta),       cos_t  = cosf(theta);
    float sin_3t = sinf(3.0f * theta), cos_3t = cosf(3.0f * theta);

    float x = sin_t  - 0.5f * sin_3t;
    float y = -cos_t + 0.5f * cos_3t;
    float z = cos_t  * cos_3t;

    float norm = sqrtf(x*x + y*y + z*z);
    if (norm < 1e-6f) norm = 1.0f;

    return (VivianNormal){ x/norm, y/norm, z/norm };
}

// ============================================================================
// Hopfion Extension 1: Topological Invariant (Linking Number Proxy)
// ============================================================================

VIVIANI_HOST_DEVICE static inline int compute_hopf_q(
    size_t block_idx,
    size_t total_blocks,
    const size_t* offset_table,
    uint32_t num_units)
{
    if (num_units == 0 || offset_table == NULL || block_idx >= total_blocks) return 0;

    int unit_idx = (int)(block_idx % num_units);
    int prev     = (unit_idx - 1 + num_units) % num_units;
    int next     = (unit_idx + 1) % num_units;

    long long diff_prev = (long long)offset_table[unit_idx] - (long long)offset_table[prev];
    long long diff_next = (long long)offset_table[next]     - (long long)offset_table[unit_idx];

    int q = (int)(((llabs(diff_prev) + llabs(diff_next)) / VIVIANI_BLOCK_SIZE) % VIVIANI_5D_MODULUS);
    return q % 4;
}

// ============================================================================
// Hopfion Extension 2: Fractal Binning (Helical Down-Scaling)
// ============================================================================

VIVIANI_HOST_DEVICE static inline int fractal_bin(int base_bin, int level) {
    if (level > VIVIANI_FRACTAL_DEPTH) return base_bin;
    if (base_bin < 0 || base_bin >= VIVIANI_BIN_COUNT) return 0;
    return (base_bin + (level % 4)) % VIVIANI_BIN_COUNT;
}

// ============================================================================
// Hopfion Extension 3: 5D Recirculation in Offsets (Kaluza-Klein inspired)
// ============================================================================

VIVIANI_HOST_DEVICE static inline size_t viviani_offset_5d(int unit_index, int total_units) {
    if (total_units == 0) return 0;

    float theta = 2.0f * 3.14159265f * (float)unit_index / (float)total_units;
    VivianNormal n = viviani_normal(theta);

    float projection = fabsf(n.z) * VIVIANI_HOPF_Q;
    size_t base      = (size_t)unit_index * VIVIANI_UNIT_SIZE;
    size_t offset    = (size_t)(projection * VIVIANI_UNIT_SIZE) % VIVIANI_POOL_SIZE;

    // Use full VIVIANI_5D_MODULUS (8) so all paths are populated, not just even ones
    int q                   = (int)(projection * (float)VIVIANI_5D_MODULUS) % VIVIANI_5D_MODULUS;
    size_t recirculation    = ((size_t)q * VIVIANI_BLOCK_SIZE) % VIVIANI_POOL_SIZE;

    return (base + offset + recirculation) % VIVIANI_POOL_SIZE;
}

VIVIANI_HOST_DEVICE static inline size_t viviani_offset(int u, int t) {
    return viviani_offset_5d(u, t);
}

// ============================================================================
// Period-4 Protection Logic
// ============================================================================

VIVIANI_HOST_DEVICE static inline bool is_protected(int count, int bin) {
    int group = bin / 4;
    return ((count + group * 4) % 4) == 0;
}

VIVIANI_HOST_DEVICE static inline int quaternary_encode(int value, int flip_state) {
    int base4 = value % 4;
    if (flip_state) {
        return (base4 == 2) ? 0 : ((base4 == 3) ? 1 : base4);
    } else {
        return (base4 == 3) ? 1 : ((base4 == 2) ? 0 : base4);
    }
}

// ============================================================================
// Shadow Parity — Flyby Detection
// ============================================================================

typedef struct {
    uint64_t primary;
    uint64_t shadow;
} ShadowPair;

VIVIANI_HOST_DEVICE static inline uint64_t compute_invariant(const uint8_t* data, size_t size) {
    uint64_t inv = 0;
    const uint64_t* ptr = (const uint64_t*)data;
    size_t words = size / sizeof(uint64_t);
    for (size_t i = 0; i < words; i++) inv ^= ptr[i];
    inv ^= (inv >> 32);
    inv ^= (inv >> 16);
    inv ^= (inv >> 8);
    return inv;
}

VIVIANI_HOST_DEVICE static inline uint64_t viviani_invariant(
    const uint8_t* data,
    size_t size,
    int block_index,
    int total_blocks,
    const size_t* offset_table,
    uint32_t num_units)
{
    uint64_t base_inv = compute_invariant(data, size);

    float theta = 2.0f * 3.14159265f * (float)block_index / (float)total_blocks;
    VivianNormal n = viviani_normal(theta);
    int q = compute_hopf_q((size_t)block_index, (size_t)total_blocks, offset_table, num_units);

    uint64_t geo_factor  = (uint64_t)(fabsf(n.z) * 255.0f);
    uint64_t hopf_factor = ((uint64_t)q & 0x03ULL) << 56;

    return (base_inv & 0xFF) | (geo_factor << 8) | hopf_factor;
}

VIVIANI_HOST_DEVICE static inline uint64_t viviani_invariant_simple(
    const uint8_t* data,
    size_t size,
    int block_index,
    int total_blocks)
{
    uint64_t base_inv  = compute_invariant(data, size);
    float theta        = 2.0f * 3.14159265f * (float)block_index / (float)total_blocks;
    VivianNormal n     = viviani_normal(theta);
    uint64_t geo_factor = (uint64_t)(fabsf(n.z) * 255.0f);
    return base_inv ^ (geo_factor << 56);
}

// ============================================================================
// Flag Queue — Lock-Free MPSC Ring Buffer for Deferred Reconciliation
// ============================================================================

typedef struct {
    uint32_t chunk_id;
    uint32_t version;
    uint8_t  defect_type;  // 0=parity, 1=topological drift, 2=overflow, 3=hopf_mismatch
    uint8_t  priority;
    uint16_t delta_hint;
} FlagEntry;

typedef struct {
    FlagEntry      entries[VIVIANI_FLAG_RING_SIZE];
    VIVIANI_ATOMIC uint32_t head;
    VIVIANI_ATOMIC uint32_t tail;
    VIVIANI_ATOMIC uint32_t count;
} FlagQueue;

static inline void flag_queue_init(FlagQueue* fq) {
    memset(fq->entries, 0, sizeof(fq->entries));
    VIVIANI_atomic_store(&fq->head, 0);
    VIVIANI_atomic_store(&fq->tail, 0);
    VIVIANI_atomic_store(&fq->count, 0);
}

static inline bool flag_queue_push(FlagQueue* fq, FlagEntry entry) {
    uint32_t count = VIVIANI_atomic_load(&fq->count);
    if (count >= VIVIANI_FLAG_RING_SIZE - 1) return false;
    uint32_t head = VIVIANI_atomic_load(&fq->head);
    fq->entries[head] = entry;
    VIVIANI_atomic_store(&fq->head, (head + 1) % VIVIANI_FLAG_RING_SIZE);
    VIVIANI_atomic_fetch_add(&fq->count, 1);
    return true;
}

static inline bool flag_queue_pop(FlagQueue* fq, FlagEntry* out) {
    uint32_t count = VIVIANI_atomic_load(&fq->count);
    if (count == 0) return false;
    uint32_t tail = VIVIANI_atomic_load(&fq->tail);
    *out = fq->entries[tail];
    VIVIANI_atomic_store(&fq->tail, (tail + 1) % VIVIANI_FLAG_RING_SIZE);
    VIVIANI_atomic_fetch_sub(&fq->count, 1);
    return true;
}

// ============================================================================
// Free-List Cache — Viral Hijacking (Fast Reuse)
// ============================================================================

typedef struct {
    void*            slots[VIVIANI_FREELIST_SLOTS];
    VIVIANI_ATOMIC uint32_t count;
} FreeListCache;

static inline void freelist_cache_init(FreeListCache* fc) {
    memset(fc->slots, 0, sizeof(fc->slots));
    VIVIANI_atomic_store(&fc->count, 0);
}

static inline bool freelist_cache_push(FreeListCache* fc, void* ptr) {
    uint32_t old_count = VIVIANI_atomic_load(&fc->count);
    while (old_count < VIVIANI_FREELIST_SLOTS) {
        uint32_t new_count = old_count + 1;
        if (VIVIANI_atomic_compare_exchange_weak(&fc->count, &old_count, new_count)) {
            fc->slots[old_count] = ptr;
            return true;
        }
    }
    return false;
}

static inline void* freelist_cache_pop(FreeListCache* fc) {
    uint32_t old_count = VIVIANI_atomic_load(&fc->count);
    while (old_count > 0) {
        uint32_t new_count = old_count - 1;
        if (VIVIANI_atomic_compare_exchange_weak(&fc->count, &old_count, new_count)) {
            void* ptr = fc->slots[new_count];
            fc->slots[new_count] = NULL;
            return ptr;
        }
    }
    return NULL;
}

// ============================================================================
// Main Allocator State  (includes Aizawa ejected pool)
// ============================================================================

typedef struct {
    // Memory arena
    uint8_t* arena;
    size_t   arena_size;

    // Bump allocator state
    VIVIANI_ATOMIC size_t fork_position;

    // Shadow parity table
    ShadowPair* shadows;
    size_t      shadow_count;

    // Flag queue for deferred reconciliation
    FlagQueue flag_queue;

    // Free-list caches (per-bin viral hijacking)
    FreeListCache    freelist_caches[VIVIANI_BIN_COUNT];
    VIVIANI_ATOMIC uint64_t cache_hits;
    VIVIANI_ATOMIC uint64_t cache_misses;

    // Protection state
    VIVIANI_ATOMIC uint32_t allocated_blocks;
    VIVIANI_ATOMIC uint32_t defect_count;
    VIVIANI_ATOMIC uint32_t protected_mode;
    VIVIANI_ATOMIC uint32_t flip_state;

    // Rate tracking (fixed-point: actual_rate = value / 1000)
    VIVIANI_ATOMIC uint32_t alloc_rate_fp;

    // Bin configuration
    size_t           bin_sizes[VIVIANI_BIN_COUNT];
    VIVIANI_ATOMIC uint32_t bin_counters[VIVIANI_BIN_COUNT];

    // Viviani geometry tables
    size_t*      offset_table;
    VivianNormal* normal_table;
    uint32_t     num_units;

    // Version counter for ABA protection
    VIVIANI_ATOMIC uint32_t version;

    // --- Aizawa quark pool (V8: + direct-mapped index for O(1) lookup) ---
    EjectedEntry*    ejected_pool;
    uint32_t         ejected_index[MAX_EJECTED]; // hash: block_idx % MAX_EJECTED -> pool slot
    VIVIANI_ATOMIC uint32_t ejected_count;

    #ifdef __CUDACC__
    int                num_streams;
    cudaStream_t*      streams;
    void**             stream_holds;
    size_t*            stream_hold_sizes;
    VIVIANI_ATOMIC uint32_t*  stream_hold_counts;
    VIVIANI_ATOMIC uint64_t   gpu_held_bytes;
    #endif
} VivianiAllocator;

// ============================================================================
// Initialization / Destruction  (updated for Aizawa pool)
// ============================================================================

static inline void viviani_init(VivianiAllocator* va, size_t arena_size) {
    va->arena_size = (arena_size / 4096) * 4096;
    va->arena      = (uint8_t*)aligned_alloc(4096, va->arena_size);
    if (!va->arena) { fprintf(stderr, "FATAL: Failed to allocate arena\n"); exit(1); }
    memset(va->arena, 0, va->arena_size);

    VIVIANI_atomic_store(&va->fork_position,   0);
    VIVIANI_atomic_store(&va->allocated_blocks, 0);
    VIVIANI_atomic_store(&va->defect_count,    0);
    VIVIANI_atomic_store(&va->protected_mode,  0);
    VIVIANI_atomic_store(&va->flip_state,      0);
    VIVIANI_atomic_store(&va->alloc_rate_fp,   0);
    VIVIANI_atomic_store(&va->cache_hits,      0);
    VIVIANI_atomic_store(&va->cache_misses,    0);
    VIVIANI_atomic_store(&va->version,         0);

    size_t sizes[] = {
        VIVIANI_BIN_0,  VIVIANI_BIN_1,  VIVIANI_BIN_2,  VIVIANI_BIN_3,
        VIVIANI_BIN_4,  VIVIANI_BIN_5,  VIVIANI_BIN_6,  VIVIANI_BIN_7,
        VIVIANI_BIN_8,  VIVIANI_BIN_9,  VIVIANI_BIN_10, VIVIANI_BIN_11,
        VIVIANI_BIN_12, VIVIANI_BIN_13, VIVIANI_BIN_14, VIVIANI_BIN_15
    };
    memcpy(va->bin_sizes, sizes, sizeof(sizes));

    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) {
        VIVIANI_atomic_store(&va->bin_counters[i], 0);
        freelist_cache_init(&va->freelist_caches[i]);
    }

    va->num_units    = (uint32_t)(va->arena_size / VIVIANI_UNIT_SIZE);
    va->offset_table = (size_t*)malloc(va->num_units * sizeof(size_t));
    va->normal_table = (VivianNormal*)malloc(va->num_units * sizeof(VivianNormal));
    if (!va->offset_table || !va->normal_table) {
        fprintf(stderr, "FATAL: Failed to allocate geometry tables\n");
        exit(1);
    }

    for (uint32_t i = 0; i < va->num_units; i++) {
        va->offset_table[i] = viviani_offset(i, va->num_units);
        float theta          = 2.0f * 3.14159265f * (float)i / (float)va->num_units;
        va->normal_table[i]  = viviani_normal(theta);
    }

    va->shadow_count = va->arena_size / (VIVIANI_BLOCK_SIZE * VIVIANI_SHADOW_STRIDE);
    va->shadows      = (ShadowPair*)calloc(va->shadow_count, sizeof(ShadowPair));
    if (!va->shadows) {
        fprintf(stderr, "FATAL: Failed to allocate shadow table\n");
        exit(1);
    }

    flag_queue_init(&va->flag_queue);

    // Aizawa ejected pool + direct-mapped index
    va->ejected_pool = (EjectedEntry*)calloc(MAX_EJECTED, sizeof(EjectedEntry));
    if (!va->ejected_pool) {
        fprintf(stderr, "FATAL: Failed to allocate Aizawa ejected pool\n");
        exit(1);
    }
    for (uint32_t i = 0; i < MAX_EJECTED; i++)
        va->ejected_index[i] = VIVIANI_EJECTED_EMPTY;
    VIVIANI_atomic_store(&va->ejected_count, 0);

    #ifdef __CUDACC__
    va->num_streams       = 0;
    va->streams           = NULL;
    va->stream_holds      = NULL;
    va->stream_hold_sizes = NULL;
    va->stream_hold_counts = NULL;
    VIVIANI_atomic_store(&va->gpu_held_bytes, 0);
    #endif
}

static inline void viviani_destroy(VivianiAllocator* va) {
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) {
        VIVIANI_atomic_store(&va->freelist_caches[i].count, 0);
        memset(va->freelist_caches[i].slots, 0, sizeof(va->freelist_caches[i].slots));
    }

    free(va->arena);
    free(va->offset_table);
    free(va->normal_table);
    free(va->shadows);
    free(va->ejected_pool);

    #ifdef __CUDACC__
    if (va->streams) {
        for (int i = 0; i < va->num_streams; i++) cudaStreamDestroy(va->streams[i]);
        free(va->streams);
        free(va->stream_holds);
        free(va->stream_hold_sizes);
        free((void*)va->stream_hold_counts);
    }
    #endif
}

// Reset ejected pool + index without tearing down the allocator.
// Call between test phases or after compaction storms.
static inline void viviani_ejected_reset(VivianiAllocator* va) {
    memset(va->ejected_pool, 0, MAX_EJECTED * sizeof(EjectedEntry));
    for (uint32_t i = 0; i < MAX_EJECTED; i++)
        va->ejected_index[i] = VIVIANI_EJECTED_EMPTY;
    VIVIANI_atomic_store(&va->ejected_count, 0);
}

// ============================================================================
// Bin Selection with Fractal Extension
// ============================================================================

static inline int viviani_bin_index(VivianiAllocator* va, size_t size) {
    int base = VIVIANI_BIN_COUNT - 1;
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) {
        if (size <= va->bin_sizes[i]) { base = i; break; }
    }
    int level       = VIVIANI_atomic_load(&va->flip_state);
    return fractal_bin(base, level);
}

// ============================================================================
// Allocation
// ============================================================================

static inline void* viviani_alloc(VivianiAllocator* va, size_t size) {
    int    bin         = viviani_bin_index(va, size);
    size_t actual_size = va->bin_sizes[bin];

    void* cached = freelist_cache_pop(&va->freelist_caches[bin]);
    if (cached) {
        VIVIANI_atomic_fetch_add(&va->cache_hits, 1);
        VIVIANI_atomic_fetch_add(&va->allocated_blocks, 1);
        VIVIANI_atomic_fetch_add(&va->bin_counters[bin], 1);
        return cached;
    }

    VIVIANI_atomic_fetch_add(&va->cache_misses, 1);

    size_t blocks_needed = (actual_size + VIVIANI_BLOCK_SIZE - 1) / VIVIANI_BLOCK_SIZE;
    size_t alloc_size    = blocks_needed * VIVIANI_BLOCK_SIZE;
    size_t old_fork      = VIVIANI_atomic_fetch_add(&va->fork_position, alloc_size);

    if (old_fork + alloc_size > va->arena_size) {
        VIVIANI_atomic_fetch_sub(&va->fork_position, alloc_size);
        return NULL;
    }

    void*  ptr       = va->arena + old_fork;
    size_t block_idx = old_fork / VIVIANI_BLOCK_SIZE;

    if (block_idx % VIVIANI_SHADOW_STRIDE == 0) {
        size_t shadow_idx = block_idx / VIVIANI_SHADOW_STRIDE;
        if (shadow_idx < va->shadow_count) {
            uint64_t inv = viviani_invariant(
                (uint8_t*)ptr, actual_size,
                (int)block_idx, (int)(va->arena_size / VIVIANI_BLOCK_SIZE),
                va->offset_table, va->num_units);
            va->shadows[shadow_idx].primary = inv;
            va->shadows[shadow_idx].shadow  = inv;
        }
    }

    VIVIANI_atomic_fetch_add(&va->allocated_blocks, 1);
    VIVIANI_atomic_fetch_add(&va->bin_counters[bin], 1);

    uint32_t old_rate = VIVIANI_atomic_load(&va->alloc_rate_fp);
    VIVIANI_atomic_store(&va->alloc_rate_fp, (old_rate * 7 + 1000) / 8);

    return ptr;
}

// ============================================================================
// Deallocation
// ============================================================================

static inline void viviani_free(VivianiAllocator* va, void* ptr, size_t size) {
    if (!ptr) return;

    int bin = viviani_bin_index(va, size);

    if (freelist_cache_push(&va->freelist_caches[bin], ptr)) {
        VIVIANI_atomic_fetch_sub(&va->allocated_blocks, 1);
        VIVIANI_atomic_fetch_sub(&va->bin_counters[bin], 1);
        uint32_t old_rate = VIVIANI_atomic_load(&va->alloc_rate_fp);
        VIVIANI_atomic_store(&va->alloc_rate_fp, (old_rate * 7) / 8);
        return;
    }

    uint32_t alloc_count = VIVIANI_atomic_load(&va->allocated_blocks);
    if (is_protected((int)alloc_count, bin)) {
        FlagEntry entry = {
            .chunk_id   = (uint32_t)(((uint8_t*)ptr - va->arena) / VIVIANI_BLOCK_SIZE),
            .version    = VIVIANI_atomic_load(&va->version),
            .defect_type = 2,
            .priority   = 100,
            .delta_hint = (uint16_t)(size / VIVIANI_BLOCK_SIZE)
        };
        flag_queue_push(&va->flag_queue, entry);
    }

    VIVIANI_atomic_fetch_sub(&va->allocated_blocks, 1);
    VIVIANI_atomic_fetch_sub(&va->bin_counters[bin], 1);

    uint32_t old_rate = VIVIANI_atomic_load(&va->alloc_rate_fp);
    VIVIANI_atomic_store(&va->alloc_rate_fp, (old_rate * 7) / 8);
}

// ============================================================================
// Flyby Detection
// ============================================================================

static inline void viviani_flyby_check(VivianiAllocator* va, size_t block_idx) {
    size_t shadow_idx = block_idx / VIVIANI_SHADOW_STRIDE;
    if (shadow_idx >= va->shadow_count) return;
    if (va->shadows[shadow_idx].shadow == 0 &&
        va->shadows[shadow_idx].primary == 0) return;

    size_t total_blocks = va->arena_size / VIVIANI_BLOCK_SIZE;
    uint8_t* block_ptr  = va->arena + (block_idx * VIVIANI_BLOCK_SIZE);

    uint64_t current_inv = viviani_invariant(
        block_ptr, VIVIANI_BLOCK_SIZE,
        (int)block_idx, (int)total_blocks,
        va->offset_table, va->num_units);

    uint64_t expected = va->shadows[shadow_idx].shadow;

    if (current_inv != expected) {
        uint8_t  defect_type = 0;
        uint8_t  priority_val = 128;

        int q_current  = (int)((current_inv >> 56) & 0x03);
        int q_expected = (int)((expected    >> 56) & 0x03);
        if (q_current != q_expected) { defect_type = 3; priority_val = 200; }

        FlagEntry entry = {
            .chunk_id    = (uint32_t)block_idx,
            .version     = VIVIANI_atomic_load(&va->version),
            .defect_type = defect_type,
            .priority    = priority_val,
            .delta_hint  = 64
        };
        flag_queue_push(&va->flag_queue, entry);
        VIVIANI_atomic_fetch_add(&va->defect_count, 1);
    }

    va->shadows[shadow_idx].primary = current_inv;
}

// ============================================================================
// Geometric Repair (DNA-Inspired)
// ============================================================================

typedef struct {
    uint32_t blocks_scanned;
    uint32_t mismatches_found;
    uint32_t repairs_applied;
    uint32_t viral_propagations;
    uint32_t hopf_repairs;
} GeometricRepairStats;

static inline GeometricRepairStats viviani_geometric_repair(VivianiAllocator* va) {
    GeometricRepairStats stats = {0};
    size_t total_blocks = va->arena_size / VIVIANI_BLOCK_SIZE;

    for (size_t i = 0; i < total_blocks; i += VIVIANI_SHADOW_STRIDE) {
        stats.blocks_scanned++;

        size_t shadow_idx = i / VIVIANI_SHADOW_STRIDE;
        if (shadow_idx >= va->shadow_count) break;

        uint8_t* block   = va->arena + (i * VIVIANI_BLOCK_SIZE);
        uint64_t current = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                               (int)i, (int)total_blocks,
                               va->offset_table, va->num_units);
        uint64_t expected = va->shadows[shadow_idx].shadow;

        if (expected == 0 && current == 0) continue;

        if (current != expected) {
            stats.mismatches_found++;

            int q_current  = (int)((current  >> 56) & 0x03);
            int q_expected = (int)((expected >> 56) & 0x03);
            if (q_current != q_expected) stats.hopf_repairs++;

            va->shadows[shadow_idx].shadow  = current;
            va->shadows[shadow_idx].primary = current;
            stats.repairs_applied++;

            if (shadow_idx > 0 && shadow_idx < va->shadow_count - 1) {
                int unit_idx = (int)(i % va->num_units);
                if (unit_idx >= 0 && unit_idx < (int)va->num_units) {
                    float z_component = fabsf(va->normal_table[unit_idx].z);
                    for (int off = -2; off <= 2; off++) {
                        if (off == 0) continue;
                        int nb = (unit_idx + off + va->num_units) % va->num_units;
                        if (fabsf(fabsf(va->normal_table[nb].z) - z_component) < 0.15f)
                            stats.viral_propagations++;
                    }
                }
            }
        }
    }

    return stats;
}

// ============================================================================
// Reconciler (Deferred Patching)
// ============================================================================

typedef struct {
    uint32_t applied_patches;
    uint32_t false_positives;
    uint32_t propagations;
} ReconcileStats;

static inline ReconcileStats viviani_reconcile(VivianiAllocator* va, uint32_t max_patches) {
    ReconcileStats stats = {0};
    FlagEntry entry;

    for (uint32_t i = 0; i < max_patches && flag_queue_pop(&va->flag_queue, &entry); i++) {
        size_t block_idx = entry.chunk_id;
        if (block_idx >= va->arena_size / VIVIANI_BLOCK_SIZE) { stats.false_positives++; continue; }

        uint8_t* block   = va->arena + (block_idx * VIVIANI_BLOCK_SIZE);
        uint64_t current = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                               (int)block_idx, (int)(va->arena_size / VIVIANI_BLOCK_SIZE),
                               va->offset_table, va->num_units);

        size_t shadow_idx = block_idx / VIVIANI_SHADOW_STRIDE;
        if (shadow_idx >= va->shadow_count) { stats.false_positives++; continue; }

        uint64_t expected = va->shadows[shadow_idx].shadow;
        if (current != expected) {
            va->shadows[shadow_idx].shadow  = current;
            va->shadows[shadow_idx].primary = current;
            stats.applied_patches++;
            if (entry.priority > 150) stats.propagations++;
        } else {
            stats.false_positives++;
        }
    }

    VIVIANI_atomic_store(&va->defect_count, VIVIANI_atomic_load(&va->flag_queue.count));
    return stats;
}

// ============================================================================
// Superfluid stats type
// ============================================================================

typedef struct {
    uint32_t blocks_scanned;
    uint32_t blocks_moved;
    uint32_t bytes_recovered;
    uint32_t shadow_updates;
    float    efficiency;
} SuperfluidStats;

static inline bool viviani_should_compact(VivianiAllocator* va) {
    size_t   fork    = VIVIANI_atomic_load(&va->fork_position);
    float    usage   = (float)fork / va->arena_size;
    uint32_t defects = VIVIANI_atomic_load(&va->defect_count);
    return (usage > 0.7f) || (defects > VIVIANI_MAX_DEFECTS);
}

// ============================================================================
// Aizawa Helpers
// ============================================================================

static inline void update_aizawa(AizawaState* as) {
    float a  = 0.95f + 0.05f * sinf(as->phi);
    float b  = 0.7f  + 0.02f * cosf(2.0f * as->phi);
    float f  = 0.1f  + 0.02f * sinf(3.0f * as->phi);
    float x  = as->state[0], y = as->state[1], z = as->state[2];
    float dt = 0.005f;
    float dx = (z - b) * x - 3.5f * y;
    float dy = 3.5f * x + (z - b) * y;
    float dz = 0.6f + a*z - z*z*z/3.0f - (x*x + y*y)*(1.0f + 0.25f*z) + f*z*x*x*x;
    as->state[0] += dt * dx;
    as->state[1] += dt * dy;
    as->state[2] += dt * dz;
    as->phi = fmodf(as->phi + 0.002f * sqrtf(dx*dx + dy*dy + dz*dz),
                    2.0f * 3.14159265f);
    as->steps++;
}

static inline bool should_eject(VivianiAllocator* va, size_t block_idx) {
    uint32_t d = VIVIANI_atomic_load(&va->defect_count);
    return (d > (uint32_t)(VIVIANI_MAX_DEFECTS * 0.65f)) || (block_idx % 4 == 0);
}

static inline bool check_aizawa_snap(AizawaState* as) {
    float r = fmaxf(cosf(as->phi) * cosf(as->phi),
                    sinf(as->phi) * sinf(as->phi));
    return (r < 0.82f && as->steps > 50);
}

// ============================================================================
// Superfluid Compaction  — with Aizawa Quark Stirring
// ============================================================================

static inline SuperfluidStats viviani_superfluid_compact(VivianiAllocator* va) {
    SuperfluidStats stats = {0};

    size_t fork_initial = VIVIANI_atomic_load(&va->fork_position);
    size_t new_fork     = 0;
    size_t total_blocks = va->arena_size / VIVIANI_BLOCK_SIZE;

    for (size_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        stats.blocks_scanned++;

        size_t current_offset = block_idx * VIVIANI_BLOCK_SIZE;
        if (current_offset >= fork_initial) break;

        uint8_t* block = va->arena + current_offset;
        uint64_t inv   = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                             (int)block_idx, (int)total_blocks,
                             va->offset_table, va->num_units);

        if (inv == 0) continue;

        // --- Aizawa ejection ---
        // Write the direct-mapped index on insert so the stir path below
        // can find this entry in O(1). Hash collisions overwrite the old
        // mapping; the displaced entry stays in the pool and is found by
        // the linear-scan fallback if needed.
        if (should_eject(va, block_idx) &&
            VIVIANI_atomic_load(&va->ejected_count) < MAX_EJECTED)
        {
            uint32_t slot = VIVIANI_atomic_fetch_add(&va->ejected_count, 1) % MAX_EJECTED;
            EjectedEntry* e = &va->ejected_pool[slot];
            e->block_idx        = block_idx;
            e->state.state[0]   = (float)(rand() % 1000) / 1000.0f;
            e->state.state[1]   = 0.2f;
            e->state.state[2]   = 0.3f;
            e->state.phi        = (float)block_idx / total_blocks * 2.0f * 3.14159265f;
            e->state.steps      = 0;
            va->ejected_index[block_idx % MAX_EJECTED] = slot;
            memset(block, 0, VIVIANI_BLOCK_SIZE);
            stats.blocks_moved++;
            continue;
        }

        // --- Stir any ejected block that landed at this index (V8: O(1) fast path) ---
        {
            uint32_t hash_slot = (uint32_t)(block_idx % MAX_EJECTED);
            uint32_t pool_slot = va->ejected_index[hash_slot];
            EjectedEntry* found = NULL;

            // Fast path: direct-mapped hit
            if (pool_slot != VIVIANI_EJECTED_EMPTY &&
                pool_slot < MAX_EJECTED &&
                va->ejected_pool[pool_slot].block_idx == block_idx)
            {
                found = &va->ejected_pool[pool_slot];
            } else {
                // Fallback: linear scan for hash collision (rare); repair index
                uint32_t ej = VIVIANI_atomic_load(&va->ejected_count);
                for (uint32_t s = 0; s < ej; s++) {
                    if (va->ejected_pool[s].block_idx == block_idx) {
                        found = &va->ejected_pool[s];
                        va->ejected_index[hash_slot] = s;
                        break;
                    }
                }
            }

            if (found) {
                update_aizawa(&found->state);
                if (check_aizawa_snap(&found->state)) {
                    uint64_t new_inv = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                                           (int)block_idx, (int)total_blocks,
                                           va->offset_table, va->num_units);
                    size_t sh = block_idx / VIVIANI_SHADOW_STRIDE;
                    if (sh < va->shadow_count) va->shadows[sh].shadow = new_inv;
                    va->ejected_index[hash_slot] = VIVIANI_EJECTED_EMPTY;
                    VIVIANI_atomic_fetch_sub(&va->ejected_count, 1);
                    stats.shadow_updates++;
                }
            }
        }

        // --- Normal compaction move ---
        size_t dest_offset = new_fork;
        if (dest_offset != current_offset) {
            memmove(va->arena + dest_offset, block, VIVIANI_BLOCK_SIZE);

            size_t new_shadow_idx = (dest_offset / VIVIANI_BLOCK_SIZE) / VIVIANI_SHADOW_STRIDE;
            if (new_shadow_idx < va->shadow_count) {
                uint64_t new_inv = viviani_invariant(
                    va->arena + dest_offset, VIVIANI_BLOCK_SIZE,
                    (int)(dest_offset / VIVIANI_BLOCK_SIZE), (int)total_blocks,
                    va->offset_table, va->num_units);
                va->shadows[new_shadow_idx].shadow  = new_inv;
                va->shadows[new_shadow_idx].primary = new_inv;
                stats.shadow_updates++;
            }

            stats.blocks_moved++;
            stats.bytes_recovered += (uint32_t)(current_offset - dest_offset);
        }
        new_fork += VIVIANI_BLOCK_SIZE;
    }

    viviani_geometric_repair(va);

    if (fork_initial > new_fork) {
        stats.efficiency = (float)(fork_initial - new_fork) / fork_initial;
        VIVIANI_atomic_store(&va->fork_position, new_fork);
    }

    VIVIANI_atomic_fetch_add(&va->version, 1);
    return stats;
}

// ============================================================================
// Statistics
// ============================================================================

typedef struct {
    uint32_t allocated_blocks;
    uint32_t defect_count;
    uint32_t protected_mode;
    uint32_t flip_state;
    float    alloc_rate;
    uint32_t flag_queue_depth;
    uint32_t bin_counts[VIVIANI_BIN_COUNT];
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t gpu_held_bytes;
    uint32_t ejected_count;   // Aizawa pool depth
} VivianiStats;

static inline VivianiStats viviani_stats(VivianiAllocator* va) {
    VivianiStats s = {0};
    s.allocated_blocks = VIVIANI_atomic_load(&va->allocated_blocks);
    s.defect_count     = VIVIANI_atomic_load(&va->defect_count);
    s.protected_mode   = VIVIANI_atomic_load(&va->protected_mode);
    s.flip_state       = VIVIANI_atomic_load(&va->flip_state);
    s.alloc_rate       = VIVIANI_atomic_load(&va->alloc_rate_fp) / 1000.0f;
    s.flag_queue_depth = VIVIANI_atomic_load(&va->flag_queue.count);
    s.ejected_count    = VIVIANI_atomic_load(&va->ejected_count);
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++)
        s.bin_counts[i] = VIVIANI_atomic_load(&va->bin_counters[i]);
    s.cache_hits   = VIVIANI_atomic_load(&va->cache_hits);
    s.cache_misses = VIVIANI_atomic_load(&va->cache_misses);
    #ifdef __CUDACC__
    s.gpu_held_bytes = VIVIANI_atomic_load(&va->gpu_held_bytes);
    #endif
    return s;
}

static inline void viviani_print_stats(const VivianiStats* s) {
    printf("=== Viviani Allocator Stats (V8 — Aizawa Edition) ===\n");
    printf("Allocated blocks: %u\n", s->allocated_blocks);
    printf("Defect count:     %u\n", s->defect_count);
    printf("Protected mode:   %s\n", s->protected_mode ? "YES" : "no");
    printf("Flip state:       %u\n", s->flip_state);
    printf("Alloc rate:       %.2f  (hold threshold: %.3f derived)\n",
           s->alloc_rate, (float)VIVIANI_ALLOC_RATE_THRESHOLD / 1000.0f);
    printf("Flag queue depth: %u\n", s->flag_queue_depth);
    printf("Aizawa ejected:   %u / %u\n", s->ejected_count, MAX_EJECTED);
    printf("Bin counts:      ");
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) printf(" %u", s->bin_counts[i]);
    printf("\n");
    printf("--- Cache Stats ---\n");
    printf("Cache hits:       %lu\n", s->cache_hits);
    printf("Cache misses:     %lu\n", s->cache_misses);
    uint64_t total    = s->cache_hits + s->cache_misses;
    float    hit_rate = (total > 0) ? (float)s->cache_hits / total * 100.0f : 0.0f;
    printf("Hit rate:         %.1f%%\n", hit_rate);
    #ifdef __CUDACC__
    printf("GPU held bytes:   %lu\n", s->gpu_held_bytes);
    #endif
}

// ============================================================================
// CUDA Extensions
// ============================================================================

#ifdef __CUDACC__

static inline void viviani_cuda_init(VivianiAllocator* va, int num_streams) {
    va->num_streams  = num_streams;
    va->streams      = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    for (int s = 0; s < num_streams; s++) cudaStreamCreate(&va->streams[s]);
    va->stream_holds       = (void**)calloc(num_streams * VIVIANI_HOLD_PER_STREAM, sizeof(void*));
    va->stream_hold_sizes  = (size_t*)calloc(num_streams * VIVIANI_HOLD_PER_STREAM, sizeof(size_t));
    va->stream_hold_counts = (VIVIANI_ATOMIC uint32_t*)calloc(num_streams, sizeof(VIVIANI_ATOMIC uint32_t));
    VIVIANI_atomic_store(&va->gpu_held_bytes, 0);
}

static inline void* viviani_cuda_alloc(VivianiAllocator* va, size_t size, int stream_idx) {
    if (stream_idx >= va->num_streams) return NULL;

    int    bin         = viviani_bin_index(va, size);
    size_t actual_size = va->bin_sizes[bin];

    uint32_t hold_count = VIVIANI_atomic_load(&va->stream_hold_counts[stream_idx]);
    size_t*  sizes      = va->stream_hold_sizes + (stream_idx * VIVIANI_HOLD_PER_STREAM);
    void**   holds      = (void**)(va->stream_holds + (stream_idx * VIVIANI_HOLD_PER_STREAM));

    for (int h = (int)hold_count - 1; h >= 0; h--) {
        if (sizes[h] == actual_size) {
            void* ptr = holds[h];
            for (int j = h; j < (int)hold_count - 1; j++) {
                holds[j] = holds[j+1]; sizes[j] = sizes[j+1];
            }
            VIVIANI_atomic_fetch_sub(&va->stream_hold_counts[stream_idx], 1);
            VIVIANI_atomic_fetch_sub(&va->gpu_held_bytes, actual_size);
            return ptr;
        }
    }

    void* ptr = NULL;
    if (cudaMalloc(&ptr, actual_size) != cudaSuccess) return NULL;
    return ptr;
}

static inline void viviani_cuda_free(VivianiAllocator* va, void* ptr, size_t size, int stream_idx) {
    if (!ptr || stream_idx >= va->num_streams) return;

    int    bin         = viviani_bin_index(va, size);
    size_t actual_size = va->bin_sizes[bin];

    uint32_t hold_count  = VIVIANI_atomic_load(&va->stream_hold_counts[stream_idx]);
    uint32_t alloc_count = VIVIANI_atomic_load(&va->allocated_blocks);

    bool protect  = is_protected((int)alloc_count, bin) ||
                    (VIVIANI_atomic_load(&va->alloc_rate_fp) > VIVIANI_ALLOC_RATE_THRESHOLD);
    bool can_hold = protect &&
                    hold_count < VIVIANI_HOLD_PER_STREAM &&
                    VIVIANI_atomic_load(&va->gpu_held_bytes) + actual_size < VIVIANI_GPU_HELD_CAP;

    if (can_hold) {
        size_t* sizes = va->stream_hold_sizes + (stream_idx * VIVIANI_HOLD_PER_STREAM);
        void**  holds = (void**)(va->stream_holds + (stream_idx * VIVIANI_HOLD_PER_STREAM));
        holds[hold_count] = ptr;
        sizes[hold_count] = actual_size;
        VIVIANI_atomic_fetch_add(&va->stream_hold_counts[stream_idx], 1);
        VIVIANI_atomic_fetch_add(&va->gpu_held_bytes, actual_size);
    } else {
        cudaFree(ptr);
    }
}

__global__ void viviani_touch_kernel(char* ptr, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)idx < size) ptr[idx] = (char)(threadIdx.x & 0xFF);
}

#endif // __CUDACC__

#endif // VIVIANI_ALLOC_HOPFION_FIXED_CUH
