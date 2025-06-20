#ifndef GPU_HMAC_CUH
#define GPU_HMAC_CUH

#include "GPUSHA512.cuh"

typedef std::vector<unsigned char> ByteVec;

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void hmac_sha512_kernel(
    const BYTE* const* keys, const size_t* key_lens,
    const BYTE* const* datas, const size_t* data_lens,
    BYTE* outputs, int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const BYTE* key = keys[idx];
    size_t key_len = key_lens[idx];
    const BYTE* data = datas[idx];
    size_t data_len = data_lens[idx];
    BYTE* out = outputs + idx * SHA512_DIGEST_SIZE;

    BYTE k[SHA512_BLOCK_SIZE] = {0};
    BYTE k_ipad[SHA512_BLOCK_SIZE];
    BYTE k_opad[SHA512_BLOCK_SIZE];
    BYTE inner_hash[SHA512_DIGEST_SIZE];

    if (key_len > SHA512_BLOCK_SIZE) {
        SHA512_CTX ctx;
        sha512_init(&ctx);
        sha512_update(&ctx, key, key_len);
        sha512_final(&ctx, k);
        key_len = SHA512_DIGEST_SIZE;
    } else {
        memcpy(k, key, key_len);
    }

    for (int i = 0; i < SHA512_BLOCK_SIZE; i++) {
        k_ipad[i] = k[i] ^ 0x36;
        k_opad[i] = k[i] ^ 0x5c;
    }

    SHA512_CTX ctx_in;
    sha512_init(&ctx_in);
    sha512_update(&ctx_in, k_ipad, SHA512_BLOCK_SIZE);
    sha512_update(&ctx_in, data, data_len);
    sha512_final(&ctx_in, inner_hash);

    SHA512_CTX ctx_out;
    sha512_init(&ctx_out);
    sha512_update(&ctx_out, k_opad, SHA512_BLOCK_SIZE);
    sha512_update(&ctx_out, inner_hash, SHA512_DIGEST_SIZE);
    sha512_final(&ctx_out, out);
}

__host__ void hmac_sha512_batch(
    const std::vector<std::string>& keys,
    const std::vector<ByteVec>& datas,
    std::vector<ByteVec>& outputs,
    int threads_per_block
) {
    int count = keys.size();
    outputs.resize(count);

    std::vector<const BYTE*> h_keys(count);
    std::vector<const BYTE*> h_datas(count);
    std::vector<size_t> h_key_lens(count), h_data_lens(count);

    BYTE** d_keys = nullptr;
    BYTE** d_datas = nullptr;
    size_t* d_key_lens = nullptr;
    size_t* d_data_lens = nullptr;
    BYTE* d_outputs = nullptr;

    std::vector<BYTE*> d_key_ptrs(count);
    std::vector<BYTE*> d_data_ptrs(count);

    // === 分配 key/data 到 device ===
    for (int i = 0; i < count; ++i) {
        h_key_lens[i] = keys[i].size();
        h_data_lens[i] = datas[i].size();

        // 每个 key
        CudaSafeCall(cudaMalloc(&d_key_ptrs[i], h_key_lens[i]));
        CudaSafeCall(cudaMemcpy(d_key_ptrs[i], keys[i].data(), h_key_lens[i], cudaMemcpyHostToDevice));

        // 每个 data
        CudaSafeCall(cudaMalloc(&d_data_ptrs[i], h_data_lens[i]));
        CudaSafeCall(cudaMemcpy(d_data_ptrs[i], datas[i].data(), h_data_lens[i], cudaMemcpyHostToDevice));
    }

    // === 分配设备指针数组 ===
    CudaSafeCall(cudaMalloc(&d_keys, count * sizeof(BYTE*)));
    CudaSafeCall(cudaMalloc(&d_datas, count * sizeof(BYTE*)));
    CudaSafeCall(cudaMemcpy(d_keys, d_key_ptrs.data(), count * sizeof(BYTE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_datas, d_data_ptrs.data(), count * sizeof(BYTE*), cudaMemcpyHostToDevice));

    // === 拷贝长度数组 ===
    CudaSafeCall(cudaMalloc(&d_key_lens, count * sizeof(size_t)));
    CudaSafeCall(cudaMalloc(&d_data_lens, count * sizeof(size_t)));
    CudaSafeCall(cudaMemcpy(d_key_lens, h_key_lens.data(), count * sizeof(size_t), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_data_lens, h_data_lens.data(), count * sizeof(size_t), cudaMemcpyHostToDevice));

    // === 分配输出 buffer ===
    CudaSafeCall(cudaMalloc(&d_outputs, count * SHA512_DIGEST_SIZE));

    // === 启动 kernel ===
    int blocks = (count + threads_per_block - 1) / threads_per_block;
    hmac_sha512_kernel<<<blocks, threads_per_block>>>(
        d_keys, d_key_lens,
        d_datas, d_data_lens,
        d_outputs, count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "[!] hmac_sha512_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    CudaSafeCall(cudaDeviceSynchronize());

    // === 回传结果 ===
    for (int i = 0; i < count; ++i) {
        outputs[i].resize(SHA512_DIGEST_SIZE);
        CudaSafeCall(cudaMemcpy(outputs[i].data(), d_outputs + i * SHA512_DIGEST_SIZE, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));
    }

    // === 释放内存 ===
    for (int i = 0; i < count; ++i) {
        cudaFree(d_key_ptrs[i]);
        cudaFree(d_data_ptrs[i]);
    }
    cudaFree(d_keys);
    cudaFree(d_datas);
    cudaFree(d_key_lens);
    cudaFree(d_data_lens);
    cudaFree(d_outputs);
}


#endif // GPU_HMAC_CUH
