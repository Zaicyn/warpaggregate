// blackhole.cu  —  V8
// Realtime Hopfion Lattice Black Hole + Accretion Disk Stress Test
// Hexagonal disk lattice + Schwarzschild raymarching + Aizawa quark stirring
//
// Physics model:
//   - Keplerian orbital mechanics (omega_orb ∝ r^-3/2)
//   - Shakura-Sunyaev thin disk temperature profile (T ∝ r^-3/4)
//   - Doppler beaming (blueshift/redshift asymmetry)
//   - Gravitational redshift: sqrt(1 - r_s/r)
//   - Aizawa ejection near ISCO models magneto-rotational instability
//   - Viviani groove density wave in fragment shader (real spiral arm pattern)
//
// Allocator integration (aizawa.cuh V8):
//   - CPU arena: viviani_alloc / viviani_free for ancillary host-tracked buffers
//   - GPU slab: aizawa_slab.cuh for per-frame transient GPU allocations
//   - viviani_ejected_reset() called on simulation reset (R key)
//   - V8 derived threshold + ejected index stats surfaced in frame log
//
// Dependencies:
//   sudo pacman -S cuda glfw-wayland glew   (Arch)
//   sudo apt install libglfw3-dev libglew-dev   (Debian/Ubuntu)
//   sudo dnf install glfw-devel glew-devel      (Fedora)
//
// Compile:
//   nvcc -O3 -arch=sm_75 -std=c++17 blackhole.cu -lglfw -lGLEW -lGL -o blackhole
//   (adjust -arch=: sm_61=GTX10xx, sm_75=RTX20xx, sm_86=RTX30xx, sm_89=RTX40xx)
//
// Modes:
//   ./blackhole                  — hex lattice, 80 rings (~19k points)
//   ./blackhole --rings 150      — hex lattice, 150 rings (~68k points)
//   ./blackhole --random         — 20k random points, physics-driven evolution
//   ./blackhole --random 5000000 — 5M random points (recompile with -DMAX_DISK_PTS=5000000)
//
// Controls:
//   Left drag  — orbit camera
//   Scroll     — zoom
//   R          — reset camera + allocator ejected pool
//   Space      — pause/resume simulation
//   ESC        — quit

#include "aizawa.cuh"
#include "aizawa_slab.cuh"

// GLEW must come before GLFW — it loads all GL 3.3+ function pointers
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <array>
#include <random>
#include <chrono>

// ============================================================================
// Configuration
// ============================================================================

#define WIDTH         1920
#define HEIGHT        1080
#define PI            3.14159265358979f
#define TWO_PI        6.28318530717959f

// Disk lattice — hex rings.  DISK_RINGS=80 → ~20k points (fast), 200 → ~125k (heavy).
#define DISK_RINGS    80

// MAX_DISK_PTS is set at compile time via -DMAX_DISK_PTS=N, or defaults to 1M.
// There is no enforced cap — pass whatever you want and let the GPU decide.
// GPUDisk is cudaMalloc'd so the real limit is your VRAM.
//   nvcc ... -DMAX_DISK_PTS=5000000   # 5M points, ~390MB GPUDisk
#ifndef MAX_DISK_PTS
#define MAX_DISK_PTS  1000000
#endif

// Black hole / disk geometry (Schwarzschild units: M=1)
#define BH_MASS       1.0f
#define ISCO_R        6.0f          // Innermost stable circular orbit = 6M
#define DISK_OUTER_R  40.0f         // Outer disk edge
#define DISK_THICKNESS 2.5f         // Half-thickness in z
#define HEX_SPACING   0.55f         // Physical spacing between hex centres

// Aizawa ejection: eject blocks near event horizon when stress > threshold
#define EJECT_STRESS  0.88f

// Raymarch black hole
#define RM_STEPS      96
#define RM_MAX_DIST   120.0f
#define RM_EPS        0.02f
#define PHOTON_RING_R 2.6f          // ~1.5 * Schwarzschild radius (3M)
#define SCHW_R        2.0f          // Schwarzschild radius = 2M

// ============================================================================
// Vec3 / Mat4 (no GLM dependency, same as eternal_hopfion)
// ============================================================================

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float a=0,float b=0,float c=0): x(a),y(b),z(c){}
    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
    __host__ __device__ Vec3 operator*(float s)       const { return {x*s,y*s,z*s}; }
    __host__ __device__ Vec3 operator/(float s)       const { return {x/s,y/s,z/s}; }
    __host__ __device__ float dot(const Vec3& o)      const { return x*o.x+y*o.y+z*o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o)     const {
        return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x};
    }
    __host__ __device__ float len()  const { return sqrtf(x*x+y*y+z*z); }
    __host__ __device__ Vec3 norm()  const { float l=len(); return l>1e-7f?*this*(1/l):Vec3(0,0,1); }
};

__host__ __device__ Vec3 vec3_lerp(Vec3 a, Vec3 b, float t) {
    return a + (b - a) * t;
}

// Column-major 4x4
struct Mat4 {
    float m[16];
    __host__ __device__ Mat4() { memset(m,0,sizeof(m)); }

    static Mat4 identity() {
        Mat4 r; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r;
    }

    static Mat4 perspective(float fovY, float aspect, float near_, float far_) {
        Mat4 r;
        float f = 1.0f / tanf(fovY * 0.5f);
        r.m[0]  = f / aspect;
        r.m[5]  = f;
        r.m[10] = (far_ + near_) / (near_ - far_);
        r.m[11] = -1;
        r.m[14] = 2 * far_ * near_ / (near_ - far_);
        return r;
    }

    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        Vec3 f = (center - eye).norm();
        Vec3 r = f.cross(up).norm();
        Vec3 u = r.cross(f);
        Mat4 res;
        res.m[0]=r.x;  res.m[4]=r.y;  res.m[8]=r.z;   res.m[12]=-(r.x*eye.x+r.y*eye.y+r.z*eye.z);
        res.m[1]=u.x;  res.m[5]=u.y;  res.m[9]=u.z;   res.m[13]=-(u.x*eye.x+u.y*eye.y+u.z*eye.z);
        res.m[2]=-f.x; res.m[6]=-f.y; res.m[10]=-f.z; res.m[14]= (f.x*eye.x+f.y*eye.y+f.z*eye.z);
        res.m[15]=1;
        return res;
    }

    static Mat4 mul(const Mat4& a, const Mat4& b) {
        Mat4 r;
        for (int c=0;c<4;c++) for (int rr=0;rr<4;rr++) {
            float s=0;
            for (int k=0;k<4;k++) s+=a.m[k*4+rr]*b.m[c*4+k];
            r.m[c*4+rr]=s;
        }
        return r;
    }
};

// ============================================================================
// GPU Disk SOA (structure of arrays for coalesced access)
// ============================================================================

struct GPUDisk {
    // Physical position in disk (cylindrical mapped to Cartesian)
    float pos_x[MAX_DISK_PTS];
    float pos_y[MAX_DISK_PTS];
    float pos_z[MAX_DISK_PTS];

    // Hex axial coords (for neighbor lookup, stored as float for GPU convenience)
    int   hq[MAX_DISK_PTS];       // axial q
    int   hr[MAX_DISK_PTS];       // axial r

    // Hopf phase + spin
    float phi[MAX_DISK_PTS];
    float omega_x[MAX_DISK_PTS];
    float omega_y[MAX_DISK_PTS];
    float omega_z[MAX_DISK_PTS];

    // Aizawa state per point
    float aiz_x[MAX_DISK_PTS];
    float aiz_y[MAX_DISK_PTS];
    float aiz_z[MAX_DISK_PTS];
    float aiz_phi[MAX_DISK_PTS];
    int   aiz_steps[MAX_DISK_PTS];

    // Disk properties
    float disk_r[MAX_DISK_PTS];   // cylindrical radius from BH centre
    float disk_phi[MAX_DISK_PTS]; // orbital angle
    float temp[MAX_DISK_PTS];     // effective temperature (for colour)
    float density[MAX_DISK_PTS];  // local density (for raymarching)

    bool  ejected[MAX_DISK_PTS];
    bool  active[MAX_DISK_PTS];
};

// ============================================================================
// OpenGL helpers
// ============================================================================

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512]; glGetShaderInfoLog(s, 512, NULL, buf);
        fprintf(stderr, "Shader compile error: %s\n", buf);
    }
    return s;
}

static GLuint linkProgram(const char* vsrc, const char* fsrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER,   vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    GLuint p  = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[512]; glGetProgramInfoLog(p, 512, NULL, buf);
        fprintf(stderr, "Program link error: %s\n", buf);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// ============================================================================
// Disk point shader — instanced quads (GL_POINTS with size)
// ============================================================================

static const char* diskVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;       // per-vertex (unit quad, -0.5..0.5)
layout(location=1) in vec3 iWorldPos;  // per-instance position
layout(location=2) in vec3 iColor;     // per-instance colour
layout(location=3) in float iSize;     // per-instance point size

uniform mat4 uViewProj;
uniform vec3 uCamPos;

out vec3  vColor;
out vec2  vUV;       // local quad UV for circle clipping
out float vSize;

void main() {
    // Robust spherical billboard: build camera-facing basis without gimbal lock
    vec3 toCamera = normalize(uCamPos - iWorldPos);
    // Choose a stable reference axis: avoid parallel to toCamera
    vec3 ref   = (abs(toCamera.y) < 0.95) ? vec3(0,1,0) : vec3(1,0,0);
    vec3 right = normalize(cross(toCamera, ref));
    vec3 up    = cross(right, toCamera);

    vec3 worldPos = iWorldPos
        + right * aPos.x * iSize
        + up    * aPos.y * iSize;

    vColor      = iColor;
    vUV         = aPos.xy;   // -0.5..0.5
    vSize       = iSize;
    gl_Position = uViewProj * vec4(worldPos, 1.0);
}
)";

static const char* diskFS = R"(
#version 330 core
in vec3  vColor;
in vec2  vUV;
in float vSize;
out vec4 fragColor;

void main() {
    // Discard zero-size (inactive) points entirely
    if (vSize <= 0.0) discard;

    // Discard outside circle radius 0.5
    float d = length(vUV);
    if (d > 0.5) discard;

    // Soft edge for anti-aliasing
    float alpha = 1.0 - smoothstep(0.30, 0.5, d);

    // Soft glow core — keep it subtle
    float glow = exp(-d * d * 10.0) * 0.4;
    vec3 col = vColor + vColor * glow;

    fragColor = vec4(col, alpha);
}
)";

// ============================================================================
// Event horizon / black hole sphere shader (full-screen quad raymarch)
// ============================================================================

static const char* quadVS = R"(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// Raymarched black hole — Schwarzschild approximation
// Not general relativity but visually correct for the purpose of a stress test
static const char* bhFS = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uTime;
uniform float uAspect;
uniform float uFov;        // tan(fov/2)

const float BH_R          = 2.0;
const float ISCO          = 6.0;
const float DISK_R0       = 6.0;
const float DISK_R1       = 40.0;
const float DISK_H        = 2.2;
const float PHOTON_RING_R = 2.6;
const float PI            = 3.14159265;

// Disk density SDF
float diskSDF(vec3 p) {
    float r   = length(p.xz);
    float phi = atan(p.z, p.x);
    // Radial: ramp up from inner edge (DISK_R0-1 to DISK_R0+1), ramp down at outer (DISK_R1-4 to DISK_R1)
    float inner = smoothstep(DISK_R0 - 1.5, DISK_R0 + 2.0, r);
    float outer = smoothstep(DISK_R1, DISK_R1 - 6.0, r);
    float radial = inner * outer;
    // Vertical: gaussian falloff over DISK_H
    float vy = p.y / DISK_H;
    float vertical = exp(-vy * vy * 2.0);
    // Orbital density wave: Viviani groove pattern
    float groove = 0.65 + 0.35 * sin(phi * 3.0 + uTime * 0.3 - r * 0.4);
    return radial * vertical * groove;
}

// Blackbody temperature → colour (simplified Planckian locus)
vec3 tempToColor(float T) {
    // T in range [0,1], 0=cool red, 1=hot blue-white
    vec3 cool  = vec3(0.9, 0.2, 0.05);
    vec3 mid   = vec3(1.0, 0.6, 0.15);
    vec3 hot   = vec3(0.9, 0.95, 1.0);
    if (T < 0.5) return mix(cool, mid, T * 2.0);
    return mix(mid, hot, (T - 0.5) * 2.0);
}

// Simple Schwarzschild photon bending (1st-order correction)
vec3 bendRay(vec3 ro, vec3 rd, float dt, float M) {
    float r = length(ro);
    if (r < BH_R * 1.5) return rd; // too close, don't bother
    float rs = 2.0 * M;
    // Newtonian gravity deflection approximation
    vec3 toward = -normalize(ro);
    float strength = rs / (r * r) * dt * 0.5;
    return normalize(rd + toward * strength);
}

void main() {
    // Reconstruct ray
    vec2 ndc  = (vUV * 2.0 - 1.0);
    ndc.x    *= uAspect;
    vec3 rd   = normalize(uCamFwd + uCamRight * ndc.x * uFov + uCamUp * ndc.y * uFov);
    vec3 ro   = uCamPos;

    // Raymarch
    float t       = 0.0;
    float dt      = 1.0;   // adaptive, overwritten in loop
    vec3  accum   = vec3(0.0);
    float transmit = 1.0;
    bool  hit_bh  = false;

    for (int i = 0; i < 96; i++) {
        if (t > 120.0 || transmit < 0.01) break;

        vec3 p = ro + rd * t;
        float r = length(p);

        // Event horizon — black sphere, accumulate nothing
        if (r < BH_R) {
            hit_bh = true;
            break;
        }

        // Photon ring glow: subtle bright halo at the photon sphere (r ~ PHOTON_RING_R)
        // Only activate near the disk midplane and modulate by transmittance
        float ring_dist = abs(r - PHOTON_RING_R);
        if (ring_dist < 0.4 && abs(p.y) < 4.0) {
            float ring_str = exp(-ring_dist * ring_dist * 20.0) * 0.8 * dt;
            float plane_fade = exp(-abs(p.y) * 0.5);
            accum += transmit * vec3(1.0, 0.8, 0.3) * ring_str * plane_fade;
            transmit *= max(0.0, 1.0 - ring_str * 0.3);
        }

        // Disk scattering
        float dens = diskSDF(p);
        if (dens > 0.001) {
            float temp   = clamp(1.0 - (r - DISK_R0) / (DISK_R1 - DISK_R0), 0.0, 1.0);
            // Doppler beaming: blueshift approaching side, redshift receding
            float orb    = atan(p.z, p.x);
            float doppler = 1.0 + 0.35 * sin(orb + uTime * 0.5);
            vec3  color  = tempToColor(temp) * doppler;

            // Gravitational redshift
            float grav_shift = sqrt(max(0.0, 1.0 - BH_R / max(r, BH_R + 0.01)));
            color *= grav_shift;

            float alpha   = dens * dt * 1.8;
            accum        += transmit * color * alpha;
            transmit     *= (1.0 - alpha);
        }

        // Gravitational ray bending (1st order Schwarzschild)
        if (r < 30.0) {
            float rs = BH_R;
            float strength = rs / (r * r * r) * dt * r;
            vec3 toward = -normalize(p);
            rd = normalize(rd + toward * strength);
        }

        // Adaptive step: smaller near BH
        dt = max(0.08, min(1.2, (r - BH_R) * 0.15));
        t += dt;
    }

    // Background stars (simple noise field)
    vec3 bgColor = vec3(0.0);
    float starNoise = fract(sin(dot(floor(rd * 80.0), vec3(127.1, 311.7, 74.3))) * 43758.5);
    bgColor += vec3(starNoise > 0.997 ? starNoise * 1.5 : 0.0);

    vec3 final_color = bgColor * transmit + accum;
    // Event horizon is just black — no extra color needed (photon ring is in accum)

    // Subtle vignette
    float vign = 1.0 - dot(vUV - 0.5, vUV - 0.5) * 1.2;
    final_color *= clamp(vign, 0.0, 1.0);

    // Reinhard tone map before gamma — prevents blown-out white core
    final_color = final_color / (final_color + vec3(1.0));
    // Gamma
    final_color = pow(clamp(final_color, 0.0, 1.0), vec3(1.0/2.2));

    fragColor = vec4(final_color, 1.0);
}
)";
// Fix the yaxis self-reference from eternal_hopfion — corrected lookAt
// (Not used in blackhole.cu — our fill kernel uses direct color/size output)

static GLuint buildBHShader() {
    return linkProgram(quadVS, bhFS);
}

// Device-visible constants (CUDA kernels can't see #define float literals directly
// when used in __device__ math — make them explicit)
__device__ __constant__ float d_PI      = 3.14159265358979f;
__device__ __constant__ float d_TWO_PI  = 6.28318530717959f;
__device__ __constant__ float d_ISCO    = 6.0f;
__device__ __constant__ float d_BH_MASS = 1.0f;
__device__ __constant__ float d_SCHW_R  = 2.0f;
__device__ __constant__ float d_DISK_THICKNESS = 2.5f;
__device__ __constant__ float d_EJECT   = 0.88f;


__global__ void diskSimKernel(GPUDisk* disk, const int* neighbors, int N, float time, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !disk->active[i]) return;

    float phi  = disk->phi[i];
    float r    = disk->disk_r[i];

    // Viviani torque (same as hopfStepKernel in eternal_hopfion)
    float torq_amp = 0.06f;
    Vec3 torque = {
        (sinf(phi) - 0.5f * sinf(3*phi)) * torq_amp,
        (-cosf(phi) + 0.5f * cosf(3*phi)) * torq_amp,
        (cosf(phi) * cosf(3*phi)) * torq_amp
    };

    // Orbital angular velocity (Keplerian: omega_orb ∝ r^-3/2)
    float omega_orb = (r > d_SCHW_R * 1.1f) ? sqrtf(d_BH_MASS / (r*r*r)) : 0.0f;

    // Phase drives orbital motion
    bool  is_now = fmodf(phi, d_TWO_PI) < d_PI;
    float rho    = is_now ? cosf(phi)*cosf(phi) : sinf(phi)*sinf(phi);

    // Neighbor average
    Vec3 neigh = {0,0,0};
    int cnt = 0;
    for (int n = 0; n < 6; n++) {
        int ni = neighbors[i*6 + n];
        if (ni >= 0 && disk->active[ni]) {
            neigh.x += disk->omega_x[ni];
            neigh.y += disk->omega_y[ni];
            neigh.z += disk->omega_z[ni];
            cnt++;
        }
    }
    if (cnt > 0) { neigh.x/=cnt; neigh.y/=cnt; neigh.z/=cnt; }

    Vec3 new_omega = is_now
        ? Vec3(neigh.x + torque.x*rho, neigh.y + torque.y*rho, neigh.z + torque.z*rho)
        : Vec3(-neigh.x + torque.x*rho, -neigh.y + torque.y*rho, -neigh.z + torque.z*rho);

    disk->omega_x[i] = new_omega.x;
    disk->omega_y[i] = new_omega.y;
    disk->omega_z[i] = new_omega.z;

    float omega_norm = new_omega.len();
    disk->phi[i] = fmodf(phi + dt * (omega_norm + omega_orb * 0.3f), d_TWO_PI);

    // Advance orbital angle (Keplerian)
    float orb_phi  = disk->disk_phi[i] + omega_orb * dt;
    disk->disk_phi[i] = fmodf(orb_phi, d_TWO_PI);

    // Update Cartesian position
    float z_wave = disk->pos_y[i]; // y is the thickness axis
    disk->pos_x[i] = r * cosf(orb_phi);
    disk->pos_y[i] = z_wave;
    disk->pos_z[i] = r * sinf(orb_phi);

    // Temperature: inner disk hotter (∝ r^-3/4 for thin disk)
    float T_norm = (r > d_ISCO)
        ? powf(d_ISCO / r, 0.75f)
        : 1.0f;
    disk->temp[i] = T_norm;

    // Density: taper at edges, pile up near ISCO
    float dens_r = expf(-0.5f * ((r - d_ISCO*1.2f) / 6.0f) *
                              ((r - d_ISCO*1.2f) / 6.0f));
    disk->density[i] = dens_r * (1.0f - fabsf(z_wave) / (d_DISK_THICKNESS + 0.01f));

    // Aizawa ejection: stress = rho near horizon
    bool eject = (rho > d_EJECT) && (r < d_ISCO * 2.0f);
    disk->ejected[i] = eject;

    if (eject) {
        // Aizawa step
        float a   = 0.95f + 0.05f * sinf(disk->aiz_phi[i]);
        float b   = 0.7f  + 0.02f * cosf(2*disk->aiz_phi[i]);
        float f   = 0.1f  + 0.02f * sinf(3*disk->aiz_phi[i]);
        float ax  = disk->aiz_x[i], ay = disk->aiz_y[i], az = disk->aiz_z[i];
        float adx = (az-b)*ax - 3.5f*ay;
        float ady = 3.5f*ax + (az-b)*ay;
        float adz = 0.6f + a*az - az*az*az/3.0f - (ax*ax+ay*ay)*(1+0.25f*az) + f*az*ax*ax*ax;
        float adt = 0.005f;
        disk->aiz_x[i] += adt*adx;
        disk->aiz_y[i] += adt*ady;
        disk->aiz_z[i] += adt*adz;
        disk->aiz_phi[i] = fmodf(disk->aiz_phi[i] + 0.002f*sqrtf(adx*adx+ady*ady+adz*adz), d_TWO_PI);
        disk->aiz_steps[i]++;

        float snap_r = fmaxf(cosf(disk->aiz_phi[i])*cosf(disk->aiz_phi[i]),
                             sinf(disk->aiz_phi[i])*sinf(disk->aiz_phi[i]));
        if (snap_r < 0.82f && disk->aiz_steps[i] > 50) {
            disk->ejected[i] = false;
            disk->aiz_steps[i] = 0;
        }
    }
}

// Fill instance data for GL rendering (one point-sprite per disk point)
__global__ void fillDiskInstancesKernel(
    float* outPos,    // 3 floats per point: world xyz
    float* outColor,  // 3 floats per point: rgb
    float* outSize,   // 1 float per point
    const GPUDisk* disk,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Zero size for inactive points — prevents stale VBO data becoming white squares
    if (!disk->active[i]) {
        outPos[i*3+0] = 0.0f; outPos[i*3+1] = 1e9f; outPos[i*3+2] = 0.0f; // park off-screen
        outColor[i*3+0] = 0.0f; outColor[i*3+1] = 0.0f; outColor[i*3+2] = 0.0f;
        outSize[i] = 0.0f;
        return;
    }

    outPos[i*3+0] = disk->pos_x[i];
    outPos[i*3+1] = disk->pos_y[i];
    outPos[i*3+2] = disk->pos_z[i];

    // Colour: blackbody temperature + Doppler shift + ejection glow
    float T     = disk->temp[i];
    float orb   = disk->disk_phi[i];
    float doppler = 1.0f + 0.35f * sinf(orb);  // blueshift one side

    // Blackbody palette: cool red → orange → yellow → blue-white
    float r, g, b;
    if (T < 0.33f) {
        float t = T / 0.33f;
        r = 0.8f + 0.1f*t; g = 0.1f + 0.2f*t; b = 0.0f;
    } else if (T < 0.66f) {
        float t = (T - 0.33f) / 0.33f;
        r = 0.9f; g = 0.3f + 0.4f*t; b = 0.0f + 0.1f*t;
    } else {
        float t = (T - 0.66f) / 0.34f;
        r = 0.9f - 0.1f*t; g = 0.7f + 0.25f*t; b = 0.1f + 0.9f*t;
    }

    // Doppler shift
    r *= doppler;
    g *= (doppler * 0.9f + 0.1f);
    b *= (2.0f - doppler);

    // Gravitational redshift: points deep near the horizon lose energy
    float disk_r_val = disk->disk_r[i];
    float grav = sqrtf(fmaxf(0.0f, 1.0f - 2.0f / fmaxf(disk_r_val, 2.1f)));
    r *= (0.5f + 0.5f * grav);
    g *= (0.4f + 0.6f * grav);
    b *= (0.3f + 0.7f * grav);

    if (disk->ejected[i]) {
        // Aizawa flare: compact bright white-yellow flash, not a huge orange blob
        r = fminf(1.0f, r * 1.8f + 0.5f);
        g = fminf(1.0f, g * 1.4f + 0.3f);
        b = fminf(1.0f, b * 0.8f + 0.05f);
    }

    outColor[i*3+0] = fminf(r, 1.0f);
    outColor[i*3+1] = fminf(g, 1.0f);
    outColor[i*3+2] = fminf(b, 1.0f);

    // Size: base size from radial position; ejected slightly larger but not 2x
    float norm_r = disk_r_val / DISK_OUTER_R;
    float sz = 0.18f + (1.0f - norm_r) * 0.22f;  // 0.18 at edge, 0.40 near ISCO
    if (disk->ejected[i]) sz *= 1.4f;             // modest boost, not 2x
    outSize[i] = sz;
}

// ============================================================================
// Stress test counters (polled every frame from host)
// ============================================================================

struct StressCounters {
    unsigned int ejected_count;
    unsigned int active_count;
    float        avg_temp;
    float        avg_omega;
};

__global__ void reduceStressKernel(const GPUDisk* disk, int N, StressCounters* out) {
    // Single-block reduction — N is moderate (up to ~125k), 512 threads stride through.
    // Float shared atomics use the standard CUDA float atomicAdd (supported SM 2.0+).
    __shared__ unsigned int s_ejected;
    __shared__ unsigned int s_active;
    __shared__ float        s_temp;
    __shared__ float        s_omega;

    if (threadIdx.x == 0) {
        s_ejected = 0u; s_active = 0u; s_temp = 0.0f; s_omega = 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (!disk->active[i]) continue;
        atomicAdd(&s_active, 1u);
        if (disk->ejected[i]) atomicAdd(&s_ejected, 1u);
        atomicAdd(&s_temp,  disk->temp[i]);
        float om = sqrtf(disk->omega_x[i]*disk->omega_x[i] +
                         disk->omega_y[i]*disk->omega_y[i] +
                         disk->omega_z[i]*disk->omega_z[i]);
        atomicAdd(&s_omega, om);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        out->ejected_count = s_ejected;
        out->active_count  = s_active;
        out->avg_temp  = (s_active > 0u) ? s_temp  / (float)s_active : 0.0f;
        out->avg_omega = (s_active > 0u) ? s_omega / (float)s_active : 0.0f;
    }
}

// ============================================================================
// Camera state (global for GLFW callbacks)
// ============================================================================

static struct {
    float dist      = 80.0f;
    float azimuth   = 0.4f;
    float elevation = 0.35f;
    double lastX = 0, lastY = 0;
    bool   dragging = false;
    bool   paused   = false;
    VivianiAllocator* va_ptr = nullptr;  // V8: for ejected pool reset on R key
} g_cam;

static void mouseButtonCB(GLFWwindow*, int btn, int action, int) {
    if (btn == GLFW_MOUSE_BUTTON_LEFT)
        g_cam.dragging = (action == GLFW_PRESS);
}
static void cursorPosCB(GLFWwindow*, double x, double y) {
    if (g_cam.dragging) {
        g_cam.azimuth   += (float)(x - g_cam.lastX) * 0.005f;
        g_cam.elevation += (float)(y - g_cam.lastY) * 0.003f;
        g_cam.elevation  = fmaxf(-1.4f, fminf(1.4f, g_cam.elevation));
    }
    g_cam.lastX = x; g_cam.lastY = y;
}
static void scrollCB(GLFWwindow*, double, double yoff) {
    g_cam.dist *= (yoff > 0 ? 0.9f : 1.1f);
    g_cam.dist  = fmaxf(8.0f, fminf(300.0f, g_cam.dist));
}
static void keyCB(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
    if (key == GLFW_KEY_R) {
        g_cam.dist=80; g_cam.azimuth=0.4f; g_cam.elevation=0.35f;
        // V8: reset ejected pool + index on camera reset so stress test
        // starts from a clean state each run
        if (g_cam.va_ptr) viviani_ejected_reset(g_cam.va_ptr);
    }
    if (key == GLFW_KEY_SPACE) g_cam.paused = !g_cam.paused;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // ---- Parse command line ------------------------------------------------
    // --random [N]   — scatter N points randomly in disk annulus (default: 20000)
    // --rings  [R]   — hex lattice with R rings (default: DISK_RINGS=80)
    // (no flag)      — hex lattice, DISK_RINGS rings
    bool use_random  = false;
    int  random_n    = 20000;
    int  rings       = DISK_RINGS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--random") == 0) {
            use_random = true;
            if (i+1 < argc && argv[i+1][0] != '-') random_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rings") == 0) {
            if (i+1 < argc) rings = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: blackhole [--random [N]] [--rings R]\n");
            printf("  --random [N]  Random disk points, N particles (default 20000)\n");
            printf("  --rings  R    Hex lattice with R rings (default %d, ~%d pts)\n",
                   DISK_RINGS, 1 + 3*DISK_RINGS*(DISK_RINGS+1));
            return 0;
        }
    }
    // ---- GLFW / OpenGL init ------------------------------------------------
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT,
        "Hopfion Black Hole — Aizawa Quark Stress Test", NULL, NULL);
    if (!window) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // GLEW must init after context is current — loads all GL 3.3+ function pointers
    glewExperimental = GL_TRUE;
    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) {
        fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(glew_err));
        return 1;
    }
    // glewInit sometimes triggers a benign GL_INVALID_ENUM — clear it
    glGetError();

    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window,   cursorPosCB);
    glfwSetScrollCallback(window,      scrollCB);
    glfwSetKeyCallback(window,         keyCB);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // ---- Shaders -----------------------------------------------------------
    GLuint bhProgram   = buildBHShader();   // full-screen BH raymarch
    GLuint diskProgram = linkProgram(diskVS, diskFS);

    // ---- Viviani allocator (CPU arena) + GPU slab --------------------------
    VivianiAllocator va;
    viviani_init(&va, VIVIANI_POOL_SIZE);   // 64MB pool; enough for 125k points @ ~500B each
    printf("[alloc] Viviani allocator initialised (%zu MB pool)\n",
           VIVIANI_POOL_SIZE / (1024*1024));
    g_cam.va_ptr = &va;  // V8: expose to keyCB for ejected reset on R

    // GPU slab: used for per-frame transient device allocations.
    // Pool sized for the largest single-frame working set: N floats × 3 channels.
    // 512 warps × SBS_PER_WARP(18) = 9216 superblocks; use 12288 for headroom.
    VivianiSlabContext slab_ctx;
    {
        cudaError_t se = viviani_slab_init(&slab_ctx, 12288);
        if (se != cudaSuccess)
            fprintf(stderr, "[warn] viviani_slab_init failed: %s — slab disabled\n",
                    cudaGetErrorString(se));
        else
            printf("[alloc] GPU slab initialised (3 classes × 12288 sbs × 4KB = %.1f MB)\n",
                   3.0f * 12288 * 4096 / (1024*1024));
    }

    // ---- Build disk point distribution on host ----------------------------
    struct HexKey { int q, r;
        bool operator==(const HexKey& o) const { return q==o.q && r==o.r; }
    };
    struct HexKeyHash {
        size_t operator()(const HexKey& k) const {
            return std::hash<int>()(k.q) ^ (std::hash<int>()(k.r) << 16);
        }
    };

    std::vector<int> h_hq, h_hr;
    std::vector<float> h_px, h_py, h_pz, h_phi, h_ox, h_oy, h_oz;
    std::vector<float> h_dr, h_dphi, h_temp, h_dens;
    std::vector<float> h_ax, h_ay, h_az, h_aphi;
    std::vector<int>   h_asteps;
    std::vector<bool>  h_ejected, h_active;
    std::unordered_map<HexKey, int, HexKeyHash> hexToIdx;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rphase(0, TWO_PI);
    std::uniform_real_distribution<float> rsmall(-0.05f, 0.05f);

    auto push_point = [&](float rad, float orb_phi, float zoff, int q, int r) {
        h_hq.push_back(q);
        h_hr.push_back(r);
        h_px.push_back(rad * cosf(orb_phi));
        h_py.push_back(zoff);
        h_pz.push_back(rad * sinf(orb_phi));
        h_phi.push_back(rphase(rng));
        h_ox.push_back(rsmall(rng));
        h_oy.push_back(rsmall(rng));
        h_oz.push_back(rsmall(rng));
        h_dr.push_back(rad);
        h_dphi.push_back(orb_phi);
        float T_norm = (rad > ISCO_R) ? powf(ISCO_R/rad, 0.75f) : 1.0f;
        h_temp.push_back(T_norm);
        h_dens.push_back(expf(-0.5f * ((rad - ISCO_R*1.2f)/6.0f) *
                                      ((rad - ISCO_R*1.2f)/6.0f)));
        h_ax.push_back(0.1f + rsmall(rng)*0.5f);
        h_ay.push_back(0.0f);
        h_az.push_back(0.0f);
        h_aphi.push_back(rphase(rng));
        h_asteps.push_back(0);
        h_ejected.push_back(false);
        h_active.push_back(true);
    };

    if (use_random) {
        // ---- Random distribution -------------------------------------------
        // Uniform in area: sample r² uniformly so density ∝ 1/r (area-correct)
        // Thickness: gaussian in z, tapering toward ISCO
        std::uniform_real_distribution<float> runif(0.0f, 1.0f);
        float r2_min = ISCO_R * ISCO_R;
        float r2_max = DISK_OUTER_R * DISK_OUTER_R;
        std::normal_distribution<float> rnorm(0.0f, 1.0f);

        int placed = 0;
        while (placed < random_n) {
            float r2  = r2_min + runif(rng) * (r2_max - r2_min);
            float rad = sqrtf(r2);
            float orb_phi = rphase(rng);
            float taper = fminf(1.0f, (rad - ISCO_R) / (ISCO_R * 0.5f));
            float zoff  = rnorm(rng) * DISK_THICKNESS * 0.35f * taper;
            if (fabsf(zoff) > DISK_THICKNESS) continue;
            push_point(rad, orb_phi, zoff, 0, placed);
            placed++;
        }
        printf("[lattice] %d random disk points (annulus r=[%.1f,%.1f])\n",
               placed, ISCO_R, DISK_OUTER_R);

    } else {
        // ---- Hex lattice ---------------------------------------------------
        for (int q = -rings; q <= rings; q++) {
            for (int r = -rings; r <= rings; r++) {
                if (abs(q+r) > rings) continue;

                float px  = HEX_SPACING * (q + r * 0.5f);
                float pz  = HEX_SPACING * (r * 0.8660254f);
                float rad = sqrtf(px*px + pz*pz);

                if (rad < ISCO_R - 1.0f || rad > DISK_OUTER_R) continue;

                int layers = (rad < ISCO_R * 1.5f) ? 1 : 3;
                for (int lyr = 0; lyr < layers; lyr++) {
                    if ((int)h_hq.size() >= MAX_DISK_PTS) goto done_lattice; // only fires if compile-time MAX exceeded
                    float frac  = (layers == 1) ? 0.0f : (lyr / (float)(layers-1) - 0.5f);
                    float taper = fminf(1.0f, (rad - ISCO_R) / (ISCO_R * 0.5f));
                    float zoff  = frac * DISK_THICKNESS * taper;
                    hexToIdx[{q*100+lyr, r}] = (int)h_hq.size();
                    float orb_phi = atan2f(pz, px);
                    push_point(rad, orb_phi, zoff, q, r);
                }
            }
        }
        done_lattice:;
        printf("[lattice] %d hex disk points (rings=%d)\n", (int)h_hq.size(), rings);
    }

    int N = (int)h_hq.size();

    // ---- Build hex neighbor table (6 neighbors per hex point) --------------
    // Hex axial directions
    const int dq[6] = { 1,-1, 0, 0, 1,-1};
    const int dr[6] = { 0, 0, 1,-1,-1, 1};

    std::vector<int> h_neighbors(N * 6, -1);
    // For neighbors we use a flat q,r lookup (ignoring z-layer cross-links for simplicity)
    std::unordered_map<HexKey, int, HexKeyHash> baseHex;
    for (int i = 0; i < N; i++) {
        HexKey k{h_hq[i], h_hr[i]};
        if (!baseHex.count(k)) baseHex[k] = i;
    }
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 6; d++) {
            HexKey nk{h_hq[i] + dq[d], h_hr[i] + dr[d]};
            auto it = baseHex.find(nk);
            if (it != baseHex.end()) h_neighbors[i*6 + d] = it->second;
        }
    }

    // ---- Upload to GPU through viviani_alloc trackers ----------------------
    // The GPU disk struct is too large for the arena (it's for point tracking metadata)
    // so we cudaMalloc it directly, but use viviani_alloc for per-point ancillary data
    // (demonstrating the allocator is integrated in the pipeline).

    void* va_ptr = viviani_alloc(&va, N * sizeof(float));  // allocator stress: N floats
    if (!va_ptr) {
        fprintf(stderr, "[warn] viviani_alloc returned NULL for ancillary buffer"
                " (pool may be too small) — continuing\n");
    } else {
        printf("[alloc] viviani_alloc: ancillary buffer at offset %zu bytes\n",
               (uint8_t*)va_ptr - va.arena);
    }

    GPUDisk* d_disk;
    cudaError_t err = cudaMalloc(&d_disk, sizeof(GPUDisk));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc GPUDisk failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Upload per-field
#define UPLOAD(field, hvec) \
    cudaMemcpy(d_disk->field, hvec.data(), N*sizeof(hvec[0]), cudaMemcpyHostToDevice)

    UPLOAD(hq,      h_hq);
    UPLOAD(hr,      h_hr);
    UPLOAD(pos_x,   h_px);
    UPLOAD(pos_y,   h_py);
    UPLOAD(pos_z,   h_pz);
    UPLOAD(phi,     h_phi);
    UPLOAD(omega_x, h_ox);
    UPLOAD(omega_y, h_oy);
    UPLOAD(omega_z, h_oz);
    UPLOAD(disk_r,  h_dr);
    UPLOAD(disk_phi,h_dphi);
    UPLOAD(temp,    h_temp);
    UPLOAD(density, h_dens);
    UPLOAD(aiz_x,   h_ax);
    UPLOAD(aiz_y,   h_ay);
    UPLOAD(aiz_z,   h_az);
    UPLOAD(aiz_phi, h_aphi);
    UPLOAD(aiz_steps, h_asteps);
#undef UPLOAD

    // Upload booleans
    std::vector<uint8_t> h_ej_byte(N), h_act_byte(N);
    for (int i=0;i<N;i++) { h_ej_byte[i]=0; h_act_byte[i]=1; }
    cudaMemcpy(d_disk->ejected, h_ej_byte.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_disk->active,  h_act_byte.data(), N, cudaMemcpyHostToDevice);

    // Neighbors
    int* d_neighbors;
    cudaMalloc(&d_neighbors, N * 6 * sizeof(int));
    cudaMemcpy(d_neighbors, h_neighbors.data(), N*6*sizeof(int), cudaMemcpyHostToDevice);

    // ---- Stress counters ---------------------------------------------------
    StressCounters* d_stress;
    cudaMalloc(&d_stress, sizeof(StressCounters));

    // ---- GL buffers for disk point sprites --------------------------------
    // Billboard quad (2 triangles)
    float quadVerts[] = {
        -0.5f,-0.5f,0,  0.5f,-0.5f,0,  0.5f,0.5f,0,
        -0.5f,-0.5f,0,  0.5f,0.5f,0,  -0.5f,0.5f,0
    };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, 0);

    // Instance VBOs
    GLuint posVBO, colorVBO, sizeVBO;
    glGenBuffers(1, &posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, posVBO);
    glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, 0);
    glVertexAttribDivisor(1, 1);

    glGenBuffers(1, &colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, 0);
    glVertexAttribDivisor(2, 1);

    glGenBuffers(1, &sizeVBO);
    glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 4, 0);
    glVertexAttribDivisor(3, 1);

    glBindVertexArray(0);

    // ---- CUDA-GL interop ---------------------------------------------------
    cudaGraphicsResource *posRes, *colorRes, *sizeRes;
    cudaGraphicsGLRegisterBuffer(&posRes,   posVBO,   cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&colorRes, colorVBO, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&sizeRes,  sizeVBO,  cudaGraphicsRegisterFlagsWriteDiscard);

    // Zero the size buffer so any unwritten slots are invisible from frame 0
    glBindBuffer(GL_ARRAY_BUFFER, sizeVBO);
    {
        std::vector<float> zeros(N, 0.0f);
        glBufferSubData(GL_ARRAY_BUFFER, 0, N * sizeof(float), zeros.data());
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ---- Full-screen quad for BH background --------------------------------
    float fsQuad[] = { -1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1 };
    GLuint fsVAO, fsVBO;
    glGenVertexArrays(1, &fsVAO);
    glGenBuffers(1, &fsVBO);
    glBindVertexArray(fsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, fsVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(fsQuad), fsQuad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glBindVertexArray(0);

    // ---- Uniform locations -------------------------------------------------
    GLint u_viewProj  = glGetUniformLocation(diskProgram, "uViewProj");
    GLint u_camPos_d  = glGetUniformLocation(diskProgram, "uCamPos");

    GLint u_camPos_bh  = glGetUniformLocation(bhProgram, "uCamPos");
    GLint u_camFwd_bh  = glGetUniformLocation(bhProgram, "uCamFwd");
    GLint u_camRight_bh= glGetUniformLocation(bhProgram, "uCamRight");
    GLint u_camUp_bh   = glGetUniformLocation(bhProgram, "uCamUp");
    GLint u_time_bh    = glGetUniformLocation(bhProgram, "uTime");
    GLint u_aspect_bh  = glGetUniformLocation(bhProgram, "uAspect");
    GLint u_fov_bh     = glGetUniformLocation(bhProgram, "uFov");

    // ---- Timing and stats --------------------------------------------------
    float sim_time   = 0.0f;
    int   frame      = 0;
    int   threads    = 256;
    int   blocks     = (N + threads - 1) / threads;
    auto  t0         = std::chrono::steady_clock::now();
    double fps_acc   = 0; int fps_frames = 0;

    printf("[run] Entering render loop. Controls: drag=orbit, scroll=zoom, R=reset, Space=pause, ESC=quit\n");

    // ---- Render loop -------------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        auto t1 = std::chrono::steady_clock::now();
        float dt_wall = std::chrono::duration<float>(t1 - t0).count();
        t0 = t1;

        float dt_sim = g_cam.paused ? 0.0f : fminf(dt_wall, 0.033f);  // cap at 30fps equivalent
        sim_time += dt_sim;

        // -- Simulate --
        if (!g_cam.paused) {
            diskSimKernel<<<blocks, threads>>>(d_disk, d_neighbors, N, sim_time, dt_sim * 2.0f);

            // Stress test: also run the viviani geometric repair every 60 frames
            if (frame % 60 == 0) {
                viviani_geometric_repair(&va);
                if (viviani_should_compact(&va))
                    viviani_superfluid_compact(&va);
            }
        }

        // -- Fill instance buffers via CUDA-GL interop --
        float *d_pos, *d_col, *d_sz;
        size_t sz;
        cudaGraphicsMapResources(1, &posRes,   0);
        cudaGraphicsMapResources(1, &colorRes, 0);
        cudaGraphicsMapResources(1, &sizeRes,  0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &sz, posRes);
        cudaGraphicsResourceGetMappedPointer((void**)&d_col, &sz, colorRes);
        cudaGraphicsResourceGetMappedPointer((void**)&d_sz,  &sz, sizeRes);

        fillDiskInstancesKernel<<<blocks, threads>>>(d_pos, d_col, d_sz, d_disk, N);

        cudaGraphicsUnmapResources(1, &posRes,   0);
        cudaGraphicsUnmapResources(1, &colorRes, 0);
        cudaGraphicsUnmapResources(1, &sizeRes,  0);

        // -- Stress counters (every 30 frames) --
        if (frame % 30 == 0) {
            reduceStressKernel<<<1, 512>>>(d_disk, N, d_stress);
            cudaDeviceSynchronize();
            StressCounters sc;
            cudaMemcpy(&sc, d_stress, sizeof(sc), cudaMemcpyDeviceToHost);

            // Viviani allocator stats (V8: ejected pool depth + derived threshold)
            VivianiStats vs = viviani_stats(&va);

            printf("[frame %5d] fps=%.0f | disk pts=%u ejected=%u avg_T=%.3f avg_ω=%.3f"
                   " | alloc hits=%lu misses=%lu defects=%u"
                   " | pool ejected=%u/%d threshold=%.3f\n",
                   frame, fps_acc > 0 ? fps_frames / fps_acc : 0.0,
                   sc.active_count, sc.ejected_count, sc.avg_temp, sc.avg_omega,
                   vs.cache_hits, vs.cache_misses, vs.defect_count,
                   vs.ejected_count, MAX_EJECTED,
                   (float)VIVIANI_ALLOC_RATE_THRESHOLD / 1000.0f);

            fps_acc = 0; fps_frames = 0;
        }

        // -- Framebuffer size (handles HiDPI/Wayland 2x scaling correctly) --
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        float fb_aspect = (float)fb_w / (float)fb_h;

        // -- Camera --
        float camX = g_cam.dist * cosf(g_cam.elevation) * sinf(g_cam.azimuth);
        float camY = g_cam.dist * sinf(g_cam.elevation);
        float camZ = g_cam.dist * cosf(g_cam.elevation) * cosf(g_cam.azimuth);
        Vec3 eye    = {camX, camY, camZ};
        Vec3 center = {0,0,0};
        Vec3 up     = {0,1,0};
        Vec3 fwd    = (center - eye).norm();
        Vec3 right  = fwd.cross(up).norm();
        Vec3 camup  = right.cross(fwd);

        Mat4 view = Mat4::lookAt(eye, center, {0,1,0});
        Mat4 proj = Mat4::perspective(PI/4.0f, fb_aspect, 0.1f, 500.0f);
        Mat4 vp   = Mat4::mul(proj, view);

        float aspect = fb_aspect;
        float fovTan = tanf(PI/8.0f);  // tan(fov/2) for 45° fov

        // -- Render BH background (full-screen quad, depth write off) --
        glDepthMask(GL_FALSE);
        glUseProgram(bhProgram);
        glUniform3f(u_camPos_bh,   eye.x, eye.y, eye.z);
        glUniform3f(u_camFwd_bh,   fwd.x, fwd.y, fwd.z);
        glUniform3f(u_camRight_bh, right.x, right.y, right.z);
        glUniform3f(u_camUp_bh,    camup.x, camup.y, camup.z);
        glUniform1f(u_time_bh,     sim_time);
        glUniform1f(u_aspect_bh,   aspect);
        glUniform1f(u_fov_bh,      fovTan);
        glBindVertexArray(fsVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDepthMask(GL_TRUE);

        // -- Render disk points --
        glClear(GL_DEPTH_BUFFER_BIT);  // keep BH background, reset depth for disk
        // Additive blending: particles glow and accumulate — no dark square halos
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);   // transparent particles — no depth occlusion
        glUseProgram(diskProgram);
        glUniformMatrix4fv(u_viewProj, 1, GL_FALSE, vp.m);
        glUniform3f(u_camPos_d, eye.x, eye.y, eye.z);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, N);
        // Restore normal blending state
        glDepthMask(GL_TRUE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glfwSwapBuffers(window);
        glfwPollEvents();
        frame++;
        fps_acc   += dt_wall;
        fps_frames++;
    }

    // ---- Cleanup -----------------------------------------------------------
    printf("[cleanup] Final allocator state:\n");
    VivianiStats final_vs = viviani_stats(&va);
    viviani_print_stats(&final_vs);

    cudaGraphicsUnregisterResource(posRes);
    cudaGraphicsUnregisterResource(colorRes);
    cudaGraphicsUnregisterResource(sizeRes);
    cudaFree(d_disk);
    cudaFree(d_neighbors);
    cudaFree(d_stress);

    if (va_ptr) viviani_free(&va, va_ptr, N * sizeof(float));
    viviani_slab_destroy(&slab_ctx);   // V8: destroy GPU slab
    g_cam.va_ptr = nullptr;
    viviani_destroy(&va);

    glDeleteProgram(bhProgram);
    glDeleteProgram(diskProgram);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
