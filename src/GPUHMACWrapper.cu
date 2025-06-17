#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "GPUHMAC.cuh"
#include "GPUSHA512.cuh"

// 常量定义
#define HMAC_BLOCK_SIZE 128
#define SHA512_DIGEST_SIZE 64

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// SHA512 主机端计算（用于 HMAC 长 key 预处理）
std::vector<unsigned char> sha512_host(const std::vector<unsigned char>& msg) {
    uint8_t *d_in = nullptr, *d_out = nullptr;
    std::vector<unsigned char> digest(SHA512_DIGEST_SIZE);

    CudaSafeCall(cudaMalloc(&d_in, msg.size()));
    CudaSafeCall(cudaMalloc(&d_out, SHA512_DIGEST_SIZE));
    CudaSafeCall(cudaMemcpy(d_in, msg.data(), msg.size(), cudaMemcpyHostToDevice));

    sha512_kernel<<<1, 1>>>(d_in, msg.size(), d_out);
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaMemcpy(digest.data(), d_out, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    return digest;
}

// HMAC-SHA512 GPU 实现（供内部或 C 接口使用）
std::vector<unsigned char> hmac_sha512_gpu(const std::vector<unsigned char>& key,
                                           const std::vector<unsigned char>& data) {
    std::vector<unsigned char> safe_key = key;
    if (safe_key.size() > HMAC_BLOCK_SIZE) {
        safe_key = sha512_host(safe_key);
    }

    const size_t key_len = safe_key.size();
    const size_t data_len = data.size();

    BYTE* d_key = nullptr;
    BYTE* d_data = nullptr;
    BYTE* d_output = nullptr;
    std::vector<unsigned char> result(SHA512_DIGEST_SIZE);

    CudaSafeCall(cudaMalloc(&d_key, key_len));
    CudaSafeCall(cudaMalloc(&d_data, data_len));
    CudaSafeCall(cudaMalloc(&d_output, SHA512_DIGEST_SIZE));

    CudaSafeCall(cudaMemcpy(d_key, safe_key.data(), key_len, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_data, data.data(), data_len, cudaMemcpyHostToDevice));

    hmac_sha512_kernel<<<1, 1>>>((const char*)d_key, key_len, d_data, data_len, d_output);
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaMemcpy(result.data(), d_output, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(d_key);
    cudaFree(d_data);
    cudaFree(d_output);

    return result;
}

__global__ void derive_masterkey_kernel(
    const char* __restrict__ mnemonics,
    const char* __restrict__ passphrases,
    const size_t* __restrict__ mnemonic_offsets,
    const size_t* __restrict__ passphrase_offsets,
    uint8_t* out_privkeys,
    uint8_t* out_chaincodes,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const char* mnemonic = mnemonics + mnemonic_offsets[idx];
    const char* passphrase = passphrases + passphrase_offsets[idx];
    uint8_t seed[64];
    uint8_t hmac_out[64];

    pbkdf2_mnemonic_to_seed(mnemonic, passphrase, seed); // GPU PBKDF2
    const char* key = "Bitcoin seed";
    hmac_sha512((const uint8_t*)key, strlen(key), seed, 64, hmac_out); // GPU 内 HMAC

    for (int i = 0; i < 32; ++i) {
        out_privkeys[idx * 32 + i] = hmac_out[i];
        out_chaincodes[idx * 32 + i] = hmac_out[i + 32];
    }
}

