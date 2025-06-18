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

#define BITCOIN_SEED "Bitcoin seed"
#define SHA512_DIGEST_SIZE 64
#define SEED_SIZE 64
#define HMAC_BLOCK_SIZE 128

// === CUDA 错误包装宏 ===
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

typedef std::vector<unsigned char> ByteVec;

__host__ std::pair<ByteVec, ByteVec> derive_master_key(const std::string& mnemonic, const std::string& passphrase) {
    initSHA512Constants();

    // 1. === 生成 seed (PBKDF2 with HMAC-SHA512) ===
    std::string salt = "mnemonic" + passphrase;
    char *d_mnemonic, *d_salt;
    BYTE *d_seed;
    CudaSafeCall(cudaMalloc(&d_mnemonic, mnemonic.size()));
    CudaSafeCall(cudaMalloc(&d_salt, salt.size()));
    CudaSafeCall(cudaMalloc(&d_seed, SEED_SIZE));

    CudaSafeCall(cudaMemcpy(d_mnemonic, mnemonic.data(), mnemonic.size(), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_salt, salt.data(), salt.size(), cudaMemcpyHostToDevice));

    pbkdf2_kernel<<<1, 1>>>(d_mnemonic, mnemonic.size(), d_salt, salt.size(), PBKDF2_HMAC_SHA512_ITERATIONS, d_seed);
    CudaSafeCall(cudaDeviceSynchronize());

    ByteVec seed(SEED_SIZE);
    CudaSafeCall(cudaMemcpy(seed.data(), d_seed, SEED_SIZE, cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaFree(d_mnemonic));
    CudaSafeCall(cudaFree(d_salt));
    CudaSafeCall(cudaFree(d_seed));

    // 2. === HMAC-SHA512(seed, key="Bitcoin seed") ===
    const char* hmac_key = BITCOIN_SEED;
    size_t key_len = strlen(hmac_key);

    BYTE *d_key, *d_data, *d_hmac_out;
    CudaSafeCall(cudaMalloc(&d_key, key_len));
    CudaSafeCall(cudaMalloc(&d_data, SEED_SIZE));
    CudaSafeCall(cudaMalloc(&d_hmac_out, SHA512_DIGEST_SIZE));

    CudaSafeCall(cudaMemcpy(d_key, hmac_key, key_len, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_data, seed.data(), SEED_SIZE, cudaMemcpyHostToDevice));

    hmac_sha512_kernel<<<1, 1>>>((const char*)d_key, key_len, d_data, SEED_SIZE, d_hmac_out);
    CudaSafeCall(cudaDeviceSynchronize());

    ByteVec I(SHA512_DIGEST_SIZE);
    CudaSafeCall(cudaMemcpy(I.data(), d_hmac_out, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaFree(d_key));
    CudaSafeCall(cudaFree(d_data));
    CudaSafeCall(cudaFree(d_hmac_out));

    if (I.size() != 64) throw std::runtime_error("HMAC result is not 64 bytes");

    return {
        ByteVec(I.begin(), I.begin() + 32), // master private key
        ByteVec(I.begin() + 32, I.end())    // chain code
    };
}

__global__ void ckd_data_kernel(
    const uint8_t* left_or_pubkey, bool hardened,
    uint32_t index, uint8_t* data_out
) {
    int offset = 0;

    if (hardened) {
        data_out[offset++] = 0x00;
        for (int i = 0; i < 32; ++i)
            data_out[offset++] = left_or_pubkey[i];
    } else {
        for (int i = 0; i < 33; ++i)
            data_out[offset++] = left_or_pubkey[i];
    }

    data_out[offset++] = (index >> 24) & 0xFF;
    data_out[offset++] = (index >> 16) & 0xFF;
    data_out[offset++] = (index >> 8) & 0xFF;
    data_out[offset++] = index & 0xFF;
}

// host端封装
std::vector<unsigned char> hmac_sha512_data(
    const std::vector<unsigned char>& key,
    const std::vector<unsigned char>& left_or_pubkey,
    bool hardened,
    uint32_t index
) {
    BYTE *d_key, *d_data, *d_out, *d_left;
    int data_len = hardened ? (1 + 32 + 4) : (33 + 4);

    // 1. GPU 内存分配
    CudaSafeCall(cudaMalloc(&d_key, key.size()));
    CudaSafeCall(cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&d_left, left_or_pubkey.size()));
    CudaSafeCall(cudaMemcpy(d_left, left_or_pubkey.data(), left_or_pubkey.size(), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&d_data, data_len));
    CudaSafeCall(cudaMalloc(&d_out, SHA512_DIGEST_SIZE));

    // 2. 构造 CKD 数据
    ckd_data_kernel<<<1, 1>>>(d_left, hardened, index, d_data);
    CudaSafeCall(cudaDeviceSynchronize());

    // 3. 执行 HMAC
    hmac_sha512_kernel<<<1, 1>>>((const char*)d_key, key.size(), d_data, data_len, d_out);
    CudaSafeCall(cudaDeviceSynchronize());

    // 4. 读取结果
    std::vector<unsigned char> result(SHA512_DIGEST_SIZE);
    CudaSafeCall(cudaMemcpy(result.data(), d_out, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));

    // 5. 清理资源
    cudaFree(d_key); cudaFree(d_left); cudaFree(d_data); cudaFree(d_out);

    return result;
}

#undef BITCOIN_SEED
#undef SHA512_DIGEST_SIZE
#undef SEED_SIZE
#undef HMAC_BLOCK_SIZE
