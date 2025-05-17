#ifndef GPU_SECP256K1_UTIL_CUH
#define GPU_SECP256K1_UTIL_CUH

#include "secp256k1.cuh"
#include "sha256.cuh"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

__device__ __forceinline__ void scalar_bigint(const uint8_t privkey[32], unsigned int out[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = ((unsigned int)privkey[i * 4 + 0] << 24) |
                 ((unsigned int)privkey[i * 4 + 1] << 16) |
                 ((unsigned int)privkey[i * 4 + 2] << 8) |
                 ((unsigned int)privkey[i * 4 + 3]);
    }
}

__device__ void scalar_multiply_unpubkey(
    const uint8_t privkey[32],
    uint8_t uncompressed_pubkey[64]
) {
    unsigned int scalar[8];
    scalar_bigint(privkey, scalar);

    unsigned int x[8], y[8];
    copyBigInt(_GX, x);
    copyBigInt(_GY, y);

    for (int i = 255; i >= 0; i--) {
        squareModP(x);
        squareModP(y);
        if ((scalar[i / 32] >> (i % 32)) & 1) {
            addModP(x, _GX, x);
            addModP(y, _GY, y);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uncompressed_pubkey[i * 4 + 0] = (x[i] >> 24) & 0xFF;
        uncompressed_pubkey[i * 4 + 1] = (x[i] >> 16) & 0xFF;
        uncompressed_pubkey[i * 4 + 2] = (x[i] >> 8) & 0xFF;
        uncompressed_pubkey[i * 4 + 3] = (x[i] >> 0) & 0xFF;
        uncompressed_pubkey[32 + i * 4 + 0] = (y[i] >> 24) & 0xFF;
        uncompressed_pubkey[32 + i * 4 + 1] = (y[i] >> 16) & 0xFF;
        uncompressed_pubkey[32 + i * 4 + 2] = (y[i] >> 8) & 0xFF;
        uncompressed_pubkey[32 + i * 4 + 3] = (y[i] >> 0) & 0xFF;
    }
}

__global__ void unpubkey_kernel(const uint8_t* privkey_in, uint8_t* pubkey_out) {
    scalar_multiply_unpubkey(privkey_in, pubkey_out);
}

inline std::vector<uint8_t> derive_unpublickey(const std::vector<uint8_t>& privkey) {
    if (privkey.size() != 32) throw std::runtime_error("Private key must be 32 bytes");

    uint8_t* d_privkey;
    uint8_t* d_pubkey;
    cudaMalloc(&d_privkey, 32);
    cudaMalloc(&d_pubkey, 64);

    cudaMemcpy(d_privkey, privkey.data(), 32, cudaMemcpyHostToDevice);
    unpubkey_kernel<<<1, 1>>>(d_privkey, d_pubkey);
    cudaDeviceSynchronize();

    std::vector<uint8_t> pubkey(64);
    cudaMemcpy(pubkey.data(), d_pubkey, 64, cudaMemcpyDeviceToHost);

    cudaFree(d_privkey);
    cudaFree(d_pubkey);

    return pubkey;
}

#endif
