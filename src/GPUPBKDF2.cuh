#ifndef PBKDF2_GPU_CUH
#define PBKDF2_GPU_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "GPUSHA512.cuh"


#define PBKDF2_HMAC_SHA512_ITERATIONS 2048
#define SEED_SIZE 64  // 512 bits
#define HMAC_BLOCK_SIZE 128  // SHA-512 block size

// Estrutura para armazenar o contexto do HMAC
struct HMAC_CTX {
    SHA512_CTX inner;
    SHA512_CTX outer;
    BYTE ipad[HMAC_BLOCK_SIZE];
    BYTE opad[HMAC_BLOCK_SIZE];
};

__device__ void hmac_sha512_init(HMAC_CTX* ctx, const BYTE* key, size_t key_len) {
    BYTE k[HMAC_BLOCK_SIZE] = {0};
    
    // Se a chave for maior que o tamanho do bloco, faça o hash dela
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

    // Prepara ipad e opad
    for (int i = 0; i < HMAC_BLOCK_SIZE; i++) {
        ctx->ipad[i] = 0x36 ^ k[i];
        ctx->opad[i] = 0x5C ^ k[i];
    }
    
    // Pré-computa os hashes iniciais do ipad e opad
    sha512_init(&ctx->inner);
    sha512_update(&ctx->inner, ctx->ipad, HMAC_BLOCK_SIZE);
    
    sha512_init(&ctx->outer);
    sha512_update(&ctx->outer, ctx->opad, HMAC_BLOCK_SIZE);
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

__global__ void pbkdf2_kernel(const char* password, size_t password_len,
                            const char* salt, size_t salt_len,
                            uint32_t iterations,
                            BYTE* output) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) { // Para seed de 512 bits, precisamos apenas de 1 bloco
        HMAC_CTX hmac_ctx;
        hmac_sha512_init(&hmac_ctx, (const BYTE*)password, password_len);
        F(&hmac_ctx, (const BYTE*)salt, salt_len, iterations, idx + 1, output + (idx * SHA512_DIGEST_SIZE));
    }
}
