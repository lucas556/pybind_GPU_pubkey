#ifndef GPU_HMAC_WRAPPER_H
#define GPU_HMAC_WRAPPER_H

#include "GPUHMAC.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

inline std::vector<unsigned char> run_hmac_sha512_gpu(const std::vector<unsigned char>& key,
                                                      const std::vector<unsigned char>& data) {
    BYTE* d_key = nullptr;
    BYTE* d_data = nullptr;
    BYTE* d_output = nullptr;

    const size_t key_len = key.size();
    const size_t data_len = data.size();
    std::vector<unsigned char> result(SHA512_DIGEST_SIZE);

    // 分配
    cudaMalloc(&d_key, key_len);
    cudaMalloc(&d_data, data_len);
    cudaMalloc(&d_output, SHA512_DIGEST_SIZE);

    // 拷贝
    cudaMemcpy(d_key, key.data(), key_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data.data(), data_len, cudaMemcpyHostToDevice);

    // 启动 kernel
    hmac_sha512_kernel<<<1, 1>>>(
        reinterpret_cast<const char*>(d_key), key_len,
        d_data, data_len,
        d_output
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    cudaMemcpy(result.data(), d_output, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_key);
    cudaFree(d_data);
    cudaFree(d_output);

    return result;
}

#endif // GPU_HMAC_WRAPPER_H
