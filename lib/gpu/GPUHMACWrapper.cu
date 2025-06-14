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

// 暴露的 C 接口（供 C++ 动态加载 .so 调用）
extern "C" {

// HMAC-SHA512 GPU 接口（外部调用）
unsigned char* hmac_sha512_interface(const unsigned char* key, size_t key_len,
                                     const unsigned char* data, size_t data_len) {
    std::vector<unsigned char> key_vec(key, key + key_len);
    std::vector<unsigned char> data_vec(data, data + data_len);
    std::vector<unsigned char> result = hmac_sha512_gpu(key_vec, data_vec);

    unsigned char* output = (unsigned char*)malloc(result.size());
    memcpy(output, result.data(), result.size());
    return output;
}

}
