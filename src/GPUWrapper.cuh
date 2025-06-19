// GPUWrapper.cu

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"
#include "GPUHMAC.cuh"
#include "constants.cuh"


#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void ckd_data_kernel_batch(
    unsigned char** pubkeys_or_lefts,  // 注意：取消 const BYTE**
    const uint8_t* hardened,
    const uint32_t* indices,
    BYTE* datas,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int offset = idx * (33 + 4);
    const BYTE* input = pubkeys_or_lefts[idx];
    BYTE* out = datas + offset;

    int pos = 0;
    if (hardened[idx]) {
        out[pos++] = 0x00;
        for (int i = 0; i < 32; ++i)
            out[pos++] = input[i];
    } else {
        for (int i = 0; i < 33; ++i)
            out[pos++] = input[i];
    }

    uint32_t index = indices[idx];
    out[pos++] = (index >> 24) & 0xFF;
    out[pos++] = (index >> 16) & 0xFF;
    out[pos++] = (index >> 8) & 0xFF;
    out[pos++] = index & 0xFF;
}

