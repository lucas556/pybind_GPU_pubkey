// File: GPUSecpBasic.h
#ifndef GPUSECPBASIC_H
#define GPUSECPBASIC_H

#include "GPUMath.h"
#include "GPUSecpHash.h"
#include <stdint.h>
#include <string.h>

// Core function to multiply privkey by generator point and return compressed pubkey
__device__ void derive_compressed_pubkey_from_private(
    const uint8_t privkey[32],
    const uint8_t* gTableX,
    const uint8_t* gTableY,
    uint8_t out[33])
{
    // Cast privkey into 16 uint16_t limbs for compatibility with _PointMultiSecp256k1
    uint16_t privKey16[16] = {0};
    for (int i = 0; i < 16; ++i) {
        privKey16[i] = ((uint16_t)privkey[2 * i] << 8) | privkey[2 * i + 1];
    }

    uint64_t qx[4] = {0};
    uint64_t qy[4] = {0};
    uint64_t qz[5] = {1, 0, 0, 0, 0};

    // Perform scalar multiplication
    _PointMultiSecp256k1(qx, qy, privKey16, (uint8_t*)gTableX, (uint8_t*)gTableY);

    // Perform modular inversion on Z and convert to affine coordinates
    _ModInv(qz);
    _ModMult(qx, qz);
    _ModMult(qy, qz);

    // Write compressed pubkey (0x02 or 0x03 + qx)
    out[0] = (qy[0] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 4; ++i) {
        out[1 + i * 8 + 0] = (qx[i] >>  0) & 0xFF;
        out[1 + i * 8 + 1] = (qx[i] >>  8) & 0xFF;
        out[1 + i * 8 + 2] = (qx[i] >> 16) & 0xFF;
        out[1 + i * 8 + 3] = (qx[i] >> 24) & 0xFF;
        out[1 + i * 8 + 4] = (qx[i] >> 32) & 0xFF;
        out[1 + i * 8 + 5] = (qx[i] >> 40) & 0xFF;
        out[1 + i * 8 + 6] = (qx[i] >> 48) & 0xFF;
        out[1 + i * 8 + 7] = (qx[i] >> 56) & 0xFF;
    }
}

#endif // GPUSECPBASIC_H
