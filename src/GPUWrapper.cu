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

__host__ std::vector<unsigned char> hmac_sha512(
    const std::vector<unsigned char>& key,
    const std::vector<unsigned char>& data,
    bool prehash_key
) {
    BYTE *d_key = nullptr, *d_data = nullptr, *d_out = nullptr;
    size_t final_key_len = key.size();

    if (prehash_key) {
        BYTE *d_tmp_in = nullptr, *d_tmp_out = nullptr;
        CudaSafeCall(cudaMalloc(&d_tmp_in, key.size()));
        CudaSafeCall(cudaMalloc(&d_tmp_out, SHA512_DIGEST_SIZE));
        CudaSafeCall(cudaMemcpy(d_tmp_in, key.data(), key.size(), cudaMemcpyHostToDevice));

        sha512_kernel<<<1, 1>>>(d_tmp_in, key.size(), d_tmp_out);
        CudaSafeCall(cudaDeviceSynchronize());

        CudaSafeCall(cudaMalloc(&d_key, SHA512_DIGEST_SIZE));
        CudaSafeCall(cudaMemcpy(d_key, d_tmp_out, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToDevice));
        final_key_len = SHA512_DIGEST_SIZE;

        CudaSafeCall(cudaFree(d_tmp_in));
        CudaSafeCall(cudaFree(d_tmp_out));
    } else {
        CudaSafeCall(cudaMalloc(&d_key, key.size()));
        CudaSafeCall(cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice));
    }

    CudaSafeCall(cudaMalloc(&d_data, data.size()));
    CudaSafeCall(cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&d_out, SHA512_DIGEST_SIZE));

    hmac_sha512_kernel<<<1, 1>>>((const char*)d_key, final_key_len, d_data, data.size(), d_out);
    CudaSafeCall(cudaDeviceSynchronize());

    std::vector<unsigned char> result(SHA512_DIGEST_SIZE);
    CudaSafeCall(cudaMemcpy(result.data(), d_out, SHA512_DIGEST_SIZE, cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaFree(d_key));
    CudaSafeCall(cudaFree(d_data));
    CudaSafeCall(cudaFree(d_out));

    return result;
}


#undef BITCOIN_SEED
#undef SHA512_DIGEST_SIZE
#undef SEED_SIZE
#undef HMAC_BLOCK_SIZE
