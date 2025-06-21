#ifndef PBKDF2_GPU_CUH
#define PBKDF2_GPU_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "GPUSHA512.cuh"

#define PBKDF2_HMAC_SHA512_ITERATIONS 2048
#define SEED_SIZE 64  // 512 bits

struct HMAC_CTX {
    SHA512_CTX inner;
    SHA512_CTX outer;
    BYTE ipad[SHA512_BLOCK_SIZE];
    BYTE opad[SHA512_BLOCK_SIZE];
};

__device__ void hmac_sha512_init(HMAC_CTX* ctx, const BYTE* key, size_t key_len) {
    BYTE k[SHA512_BLOCK_SIZE] = {0};
    if (key_len > SHA512_BLOCK_SIZE) {
        SHA512_CTX sha_ctx;
        sha512_init(&sha_ctx);
        sha512_update(&sha_ctx, key, key_len);
        sha512_final(&sha_ctx, k);
        key = k;
        key_len = SHA512_DIGEST_SIZE;
    } else {
        memcpy(k, key, key_len);
    }
    for (int i = 0; i < SHA512_BLOCK_SIZE; i++) {
        ctx->ipad[i] = 0x36 ^ k[i];
        ctx->opad[i] = 0x5C ^ k[i];
    }
    sha512_init(&ctx->inner);
    sha512_update(&ctx->inner, ctx->ipad, SHA512_BLOCK_SIZE);
    sha512_init(&ctx->outer);
    sha512_update(&ctx->outer, ctx->opad, SHA512_BLOCK_SIZE);
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

__device__ void F_threaded_multi_block(
    HMAC_CTX* ctx_base,
    const BYTE* salt, size_t salt_len,
    uint32_t iterations,
    BYTE* digest,
    uint32_t block_index
) {
    BYTE U[SHA512_DIGEST_SIZE];
    BYTE block_index_be[4];

    block_index_be[0] = (block_index >> 24) & 0xFF;
    block_index_be[1] = (block_index >> 16) & 0xFF;
    block_index_be[2] = (block_index >> 8) & 0xFF;
    block_index_be[3] = block_index & 0xFF;

    HMAC_CTX ctx = *ctx_base;
    hmac_sha512_update(&ctx, salt, salt_len);
    hmac_sha512_update(&ctx, block_index_be, 4);
    hmac_sha512_final(&ctx, U);

    for (int j = 0; j < SHA512_DIGEST_SIZE; ++j)
        digest[j] = U[j];

    for (uint32_t i = 1; i < iterations; ++i) {
        ctx = *ctx_base;
        hmac_sha512_update(&ctx, U, SHA512_DIGEST_SIZE);
        hmac_sha512_final(&ctx, U);

        #pragma unroll
        for (int j = 0; j < SHA512_DIGEST_SIZE; ++j) {
            // Warp-level XOR reduction
            BYTE val = U[j];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val ^= __shfl_xor_sync(0xffffffff, val, offset);
            }
            if ((threadIdx.x & 31) == 0) digest[j] ^= val;  // One lane per warp writes
        }
        __syncwarp();
    }
}

__global__ void pbkdf2_kernel(
    char** d_mnemonics, size_t* d_mnemonic_lens,
    char** d_salts, size_t* d_salt_lens,
    uint32_t iterations, BYTE* d_out_seeds, int count
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= count) return;

    const BYTE* password = (const BYTE*)d_mnemonics[global_idx];
    size_t password_len = d_mnemonic_lens[global_idx];
    const BYTE* salt = (const BYTE*)d_salts[global_idx];
    size_t salt_len = d_salt_lens[global_idx];
    BYTE* output = d_out_seeds + global_idx * SEED_SIZE;

    HMAC_CTX hmac_ctx;
    hmac_sha512_init(&hmac_ctx, password, password_len);

    // 单块 F 多块为 for (uint32_t i = 1; i <= Nblocks; ++i)
    F_threaded_multi_block(&hmac_ctx, salt, salt_len, iterations, output, 1);
}

#endif
