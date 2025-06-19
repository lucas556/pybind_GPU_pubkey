#ifndef PBKDF2_GPU_CUH
#define PBKDF2_GPU_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "GPUSHA512.cuh"


#define PBKDF2_HMAC_SHA512_ITERATIONS 2048
#define SEED_SIZE 64  // 512 bits

// Estrutura para armazenar o contexto do HMAC
struct HMAC_CTX {
    SHA512_CTX inner;
    SHA512_CTX outer;
    BYTE ipad[SHA512_BLOCK_SIZE];
    BYTE opad[SHA512_BLOCK_SIZE];
};

__device__ void hmac_sha512_init(HMAC_CTX* ctx, const BYTE* key, size_t key_len) {
    BYTE k[SHA512_BLOCK_SIZE] = {0};
    
    // Se a chave for maior que o tamanho do bloco, faça o hash dela
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

    // Prepara ipad e opad
    for (int i = 0; i < SHA512_BLOCK_SIZE; i++) {
        ctx->ipad[i] = 0x36 ^ k[i];
        ctx->opad[i] = 0x5C ^ k[i];
    }
    
    // Pré-computa os hashes iniciais do ipad e opad
    sha512_init(&ctx->inner);
    sha512_update(&ctx->inner, ctx->ipad, SHA512_BLOCK_SIZE);
    
    sha512_init(&ctx->outer);
    sha512_update(&ctx->outer, ctx->opad, SHA512_BLOCK_SIZE);
}

__device__ void hmac_sha512_update(HMAC_CTX* ctx, const BYTE* data, size_t len) {
    // Apenas atualiza o contexto interno
    sha512_update(&ctx->inner, data, len);
}

__device__ void hmac_sha512_final(HMAC_CTX* ctx, BYTE* output) {
    BYTE inner_hash[SHA512_DIGEST_SIZE];
    
    // Finaliza o hash interno
    SHA512_CTX inner_copy = ctx->inner;
    sha512_final(&inner_copy, inner_hash);
    
    // Copia o contexto outer pré-computado e finaliza
    SHA512_CTX outer_copy = ctx->outer;
    sha512_update(&outer_copy, inner_hash, SHA512_DIGEST_SIZE);
    sha512_final(&outer_copy, output);
}

__device__ void F(HMAC_CTX* ctx, const BYTE* salt, size_t salt_len, 
                 uint32_t iterations, uint32_t block_index, BYTE* output) {
    BYTE U[SHA512_DIGEST_SIZE];
    BYTE block_index_be[4];
    
    // Converte o índice do bloco para big-endian
    block_index_be[0] = (block_index >> 24) & 0xFF;
    block_index_be[1] = (block_index >> 16) & 0xFF;
    block_index_be[2] = (block_index >> 8) & 0xFF;
    block_index_be[3] = block_index & 0xFF;
    
    // Primeira iteração
    HMAC_CTX hmac_ctx = *ctx; // Copia o contexto pré-computado
    hmac_sha512_update(&hmac_ctx, salt, salt_len);
    hmac_sha512_update(&hmac_ctx, block_index_be, 4);
    hmac_sha512_final(&hmac_ctx, U);
    memcpy(output, U, SHA512_DIGEST_SIZE);
    
    // Iterações restantes
    for (uint32_t i = 1; i < iterations; i++) {
        hmac_ctx = *ctx; // Usa o contexto pré-computado
        hmac_sha512_update(&hmac_ctx, U, SHA512_DIGEST_SIZE);
        hmac_sha512_final(&hmac_ctx, U);
        
        for (int j = 0; j < SHA512_DIGEST_SIZE; j++) {
            output[j] ^= U[j];
        }
    }
}

__global__ void pbkdf2_kernel(
    char** d_mnemonics, size_t* d_mnemonic_lens,
    char** d_salts, size_t* d_salt_lens,
    uint32_t iterations, BYTE* d_out_seeds, int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const char* mnemonic = d_mnemonics[idx];
    size_t mnemonic_len = d_mnemonic_lens[idx];
    const char* salt = d_salts[idx];
    size_t salt_len = d_salt_lens[idx];
    BYTE* output = d_out_seeds + idx * SEED_SIZE;

    HMAC_CTX hmac_ctx;
    hmac_sha512_init(&hmac_ctx, (const BYTE*)mnemonic, mnemonic_len);
    F(&hmac_ctx, (const BYTE*)salt, salt_len, iterations, 1, output);
}
