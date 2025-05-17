#ifndef GPU_SECP256K1_UTIL_CUH
#define GPU_SECP256K1_UTIL_CUH

#include "secp256k1.cuh"
#include "sha256.cuh"

__device__ __forceinline__ void scalar_to_bigint(const uint8_t privkey[32], unsigned int out[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = ((unsigned int)privkey[i * 4 + 0] << 24) |
                 ((unsigned int)privkey[i * 4 + 1] << 16) |
                 ((unsigned int)privkey[i * 4 + 2] << 8) |
                 ((unsigned int)privkey[i * 4 + 3]);
    }
}

__device__ void scalar_multiply_and_compress_pubkey(
    const uint8_t privkey[32],
    uint8_t compressed_pubkey[33]
) {
    unsigned int scalar[8];
    scalar_to_bigint(privkey, scalar);

    unsigned int x[8], y[8];
    copyBigInt(_GX, x);
    copyBigInt(_GY, y);

    // Perform scalar multiplication via double-and-add
    for (int i = 255; i >= 0; i--) {
        squareModP(x);
        squareModP(y);
        if ((scalar[i / 32] >> (i % 32)) & 1) {
            addModP(x, _GX, x);
            addModP(y, _GY, y);
        }
    }

    // Compress public key
    compressed_pubkey[0] = (y[0] & 1) ? 0x03 : 0x02;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        compressed_pubkey[1 + i * 4 + 0] = (x[i] >> 24) & 0xFF;
        compressed_pubkey[1 + i * 4 + 1] = (x[i] >> 16) & 0xFF;
        compressed_pubkey[1 + i * 4 + 2] = (x[i] >> 8) & 0xFF;
        compressed_pubkey[1 + i * 4 + 3] = (x[i] >> 0) & 0xFF;
    }
}

#endif
