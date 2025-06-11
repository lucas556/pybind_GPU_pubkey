#ifndef GPU_HMAC_WRAPPER_H
#define GPU_HMAC_WRAPPER_H

#include "GPUHMAC.cuh"
#include "GPUSHA512.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline std::vector<unsigned char> sha512_host(const std::vector<unsigned char>& msg) {
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

/**
 * hmac_sha512_gpu
 * 用于 BIP32 CKDpriv 的子步骤（seed 派生建议使用 mnemonicToSeedGPU）。
 * key: 用于 HMAC 的 key，一般为 chain code。
 * data: HMAC 输入数据。
 * 返回值：SHA512 digest（64字节）。
 */
inline std::vector<unsigned char> hmac_sha512_gpu(const std::vector<unsigned char>& key,
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

    // 1. 分配 GPU 内存（加错误检查）
    CudaSafeCall(cudaMalloc(&d_key, key_len));
    CudaSafeCall(cudaMalloc(&d_data, data_len));
    CudaSafeCall(cudaMalloc(&d_output, SHA512_DIGEST_SIZE));

    // 2. 拷贝数据到 GPU
    CudaSafeCall(cudaMemcpy(d_key, safe_key.data(), key_len, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_data, data.data(), data_len, cudaMemcpyHostToDevice));

    // 3. 调用 GPU HMAC Kernel
    hmac_sha512_kernel<<<1, 1>>>(reinterpret_cast<const char*>(d_key), key_len,
                                 d_data, data_len, d_output);
    CudaSafeCall(cudaDeviceSynchronize());

    // 4. 拷贝结果回 Host
    CudaSafeCall(cudaMemcpy(result.data(), d_output, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));

    // 5. 清理资源
    CudaSafeCall(cudaFree(d_key));
    CudaSafeCall(cudaFree(d_data));
    CudaSafeCall(cudaFree(d_output));

    return result;
}

#endif // GPU_HMAC_WRAPPER_H
