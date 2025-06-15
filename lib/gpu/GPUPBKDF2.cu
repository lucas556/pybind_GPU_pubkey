#include "GPUPBKDF2.cuh"
#include "GPUSHA512.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>

#define PBKDF2_HMAC_SHA512_ITERATIONS 2048
#define SEED_SIZE 64
#define HMAC_BLOCK_SIZE 128

struct HMAC_CTX {
    SHA512_CTX inner;
    SHA512_CTX outer;
    BYTE ipad[HMAC_BLOCK_SIZE];
    BYTE opad[HMAC_BLOCK_SIZE];
    bool debug;
};

__device__ void hmac_sha512_init(HMAC_CTX* ctx, const BYTE* key, size_t key_len) {
    BYTE k[HMAC_BLOCK_SIZE] = {0};
    if (key_len > HMAC_BLOCK_SIZE) {
        SHA512_CTX sha_ctx;
        sha512_init(&sha_ctx);
        sha512_update(&sha_ctx, key, key_len);
        sha512_final(&sha_ctx, k);
        key = k;
        key_len = SHA512_DIGEST_SIZE;
    } else {
        memcpy(k, key, key_len);
    }
    for (int i = 0; i < HMAC_BLOCK_SIZE; i++) {
        ctx->ipad[i] = 0x36 ^ k[i];
        ctx->opad[i] = 0x5C ^ k[i];
    }
    sha512_init(&ctx->inner);
    sha512_update(&ctx->inner, ctx->ipad, HMAC_BLOCK_SIZE);
    sha512_init(&ctx->outer);
    sha512_update(&ctx->outer, ctx->opad, HMAC_BLOCK_SIZE);
}

__device__ void hmac_sha512_update(HMAC_CTX* ctx, const BYTE* data, size_t len) {
    sha512_update(&ctx->inner, data, len);
}

__device__ void hmac_sha512_final(HMAC_CTX* ctx, BYTE* output) {
    BYTE inner_hash[SHA512_DIGEST_SIZE];
    SHA512_CTX inner_copy = ctx->inner;
    sha512_final(&inner_copy, inner_hash);
    SHA512_CTX outer_copy = ctx->outer;
    sha512_update(&outer_copy, inner_hash, SHA512_DIGEST_SIZE);
    sha512_final(&outer_copy, output);
}

__device__ void F(HMAC_CTX* ctx, const BYTE* salt, size_t salt_len, 
                 uint32_t iterations, uint32_t block_index, BYTE* output) {
    BYTE U[SHA512_DIGEST_SIZE];
    BYTE block_index_be[4] = {
        (uint8_t)((block_index >> 24) & 0xFF),
        (uint8_t)((block_index >> 16) & 0xFF),
        (uint8_t)((block_index >> 8) & 0xFF),
        (uint8_t)(block_index & 0xFF)
    };
    HMAC_CTX hmac_ctx = *ctx;
    hmac_sha512_update(&hmac_ctx, salt, salt_len);
    hmac_sha512_update(&hmac_ctx, block_index_be, 4);
    hmac_sha512_final(&hmac_ctx, U);
    memcpy(output, U, SHA512_DIGEST_SIZE);

    for (uint32_t i = 1; i < iterations; i++) {
        hmac_ctx = *ctx;
        hmac_sha512_update(&hmac_ctx, U, SHA512_DIGEST_SIZE);
        hmac_sha512_final(&hmac_ctx, U);
        for (int j = 0; j < SHA512_DIGEST_SIZE; j++) {
            output[j] ^= U[j];
        }
    }
}

__global__ void pbkdf2_kernel(const char* password, size_t password_len,
                              const char* salt, size_t salt_len,
                              uint32_t iterations,
                              BYTE* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        HMAC_CTX hmac_ctx;
        hmac_ctx.debug = false;
        hmac_sha512_init(&hmac_ctx, (const BYTE*)password, password_len);
        F(&hmac_ctx, (const BYTE*)salt, salt_len, iterations, 1, output);
    }
}

std::vector<unsigned char> mnemonicToSeedGPU(const std::string& mnemonic, const std::string& passphrase) {
    std::string salt = "mnemonic" + passphrase;

    char *d_mnemonic, *d_salt;
    BYTE *d_output;

    cudaMalloc(&d_mnemonic, mnemonic.size());
    cudaMalloc(&d_salt, salt.size());
    cudaMalloc(&d_output, SEED_SIZE);

    cudaMemcpy(d_mnemonic, mnemonic.data(), mnemonic.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_salt, salt.data(), salt.size(), cudaMemcpyHostToDevice);

    pbkdf2_kernel<<<1, 1>>>(d_mnemonic, mnemonic.size(),
                            d_salt, salt.size(),
                            PBKDF2_HMAC_SHA512_ITERATIONS,
                            d_output);

    cudaDeviceSynchronize();

    std::vector<unsigned char> seed(SEED_SIZE);
    cudaMemcpy(seed.data(), d_output, SEED_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_mnemonic);
    cudaFree(d_salt);
    cudaFree(d_output);
    return seed;
}
