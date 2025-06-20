#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <secp256k1.h>
#include <iomanip>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"
#include "GPUHMAC.cuh"

namespace py = pybind11;

#define BITCOIN_SEED "Bitcoin seed"
#define MAX_CKD_DATA_SIZE 64

using ByteVec = std::vector<uint8_t>;

__host__ std::vector<unsigned char> derive_pubkey(const std::vector<unsigned char>& privkey, bool compressed = true) {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx || privkey.size() != 32 || !secp256k1_ec_seckey_verify(ctx, privkey.data()))
        throw std::runtime_error("Invalid private key");

    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privkey.data()))
        throw std::runtime_error("Pubkey creation failed");

    std::vector<unsigned char> output(compressed ? 33 : 65);
    size_t len = output.size();
    if (!secp256k1_ec_pubkey_serialize(ctx, output.data(), &len, &pubkey,
        compressed ? SECP256K1_EC_COMPRESSED : SECP256K1_EC_UNCOMPRESSED))
        throw std::runtime_error("Pubkey serialization failed");

    secp256k1_context_destroy(ctx);
    return output;
}

__global__ void ckd_data_kernel_batch(
    unsigned char** pubkeys_or_lefts,
    const uint8_t* hardened,
    const uint32_t* indices,
    BYTE* datas,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const BYTE* input = pubkeys_or_lefts[idx];
    if (!input) return;

    int offset = idx * MAX_CKD_DATA_SIZE;
    BYTE* out = datas + offset;
    int pos = 0;

    if (hardened[idx]) {
        out[pos++] = 0x00;
        for (int i = 0; i < 32; ++i)
            out[pos++] = input[i];
    } else {
        for (int i = 0; i < 33; ++i)
            out[pos++] = input[i];
    }

    uint32_t index = indices[idx];
    out[pos++] = (index >> 24) & 0xFF;
    out[pos++] = (index >> 16) & 0xFF;
    out[pos++] = (index >> 8) & 0xFF;
    out[pos++] = index & 0xFF;
} 

__host__ std::vector<unsigned char> add_privkeys_mod_n(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    std::vector<unsigned char> out = a;

    if (a.size() != 32 || b.size() != 32)
        throw std::runtime_error("Invalid key or tweak size");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx)
        throw std::runtime_error("Failed to create secp256k1 context");

    if (!secp256k1_ec_seckey_verify(ctx, b.data())) {
        secp256k1_context_destroy(ctx);
        throw std::runtime_error("Tweak IL is not a valid scalar");
    }

    if (!secp256k1_ec_seckey_tweak_add(ctx, out.data(), b.data())) {
        std::cerr << "[!] secp256k1_ec_seckey_tweak_add failed\n";
        std::cerr << "  privkey: ";
        for (unsigned char x : a) std::cerr << std::hex << std::setw(2) << std::setfill('0') << (int)x;
        std::cerr << "\n  IL: ";
        for (unsigned char x : b) std::cerr << std::hex << std::setw(2) << std::setfill('0') << (int)x;
        std::cerr << std::endl;

        secp256k1_context_destroy(ctx);
        throw std::runtime_error("secp256k1_ec_seckey_tweak_add failed");
    }

    secp256k1_context_destroy(ctx);
    return out;
}


__host__ void derive2pub(
    const std::vector<std::string>& mnemonics,
    const std::string& passphrase,
    const std::vector<uint32_t>& path_indices,
    std::vector<std::vector<unsigned char>>& final_pubkeys,
    int threads_per_block
) {
    int count = mnemonics.size();
    initSHA512Constants();

    std::vector<std::string> salts(count);
    for (int i = 0; i < count; ++i)
        salts[i] = "mnemonic" + passphrase;

    std::vector<const char*> h_mnemonics(count), h_salts(count);
    std::vector<size_t> mnemonic_lens(count), salt_lens(count);
    for (int i = 0; i < count; ++i) {
        h_mnemonics[i] = mnemonics[i].c_str();
        h_salts[i] = salts[i].c_str();
        mnemonic_lens[i] = mnemonics[i].size();
        salt_lens[i] = salts[i].size();
    }

    char** d_mnemonics; char** d_salts;
    size_t *d_mnemonic_lens, *d_salt_lens;
    BYTE* d_out_seeds;

    CudaSafeCall(cudaMalloc(&d_mnemonics, count * sizeof(char*)));
    CudaSafeCall(cudaMalloc(&d_salts, count * sizeof(char*)));
    CudaSafeCall(cudaMalloc(&d_mnemonic_lens, count * sizeof(size_t)));
    CudaSafeCall(cudaMalloc(&d_salt_lens, count * sizeof(size_t)));
    CudaSafeCall(cudaMalloc(&d_out_seeds, count * SEED_SIZE));

    std::vector<char*> d_mnemonic_data(count), d_salt_data(count);
    for (int i = 0; i < count; ++i) {
        CudaSafeCall(cudaMalloc(&d_mnemonic_data[i], mnemonic_lens[i]));
        CudaSafeCall(cudaMemcpy(d_mnemonic_data[i], h_mnemonics[i], mnemonic_lens[i], cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMalloc(&d_salt_data[i], salt_lens[i]));
        CudaSafeCall(cudaMemcpy(d_salt_data[i], h_salts[i], salt_lens[i], cudaMemcpyHostToDevice));
    }

    CudaSafeCall(cudaMemcpy(d_mnemonics, d_mnemonic_data.data(), count * sizeof(char*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_salts, d_salt_data.data(), count * sizeof(char*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_mnemonic_lens, mnemonic_lens.data(), count * sizeof(size_t), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_salt_lens, salt_lens.data(), count * sizeof(size_t), cudaMemcpyHostToDevice));

    int blocks = (count + threads_per_block - 1) / threads_per_block;
    pbkdf2_kernel<<<blocks, threads_per_block>>>(
        d_mnemonics, d_mnemonic_lens, d_salts, d_salt_lens,
        PBKDF2_HMAC_SHA512_ITERATIONS, d_out_seeds, count);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "[!] pbkdf2_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        CudaSafeCall(cudaDeviceSynchronize());
    }

    std::vector<std::vector<unsigned char>> seeds(count);
    for (int i = 0; i < count; ++i) {
        seeds[i].resize(SEED_SIZE);
        CudaSafeCall(cudaMemcpy(seeds[i].data(), d_out_seeds + i * SEED_SIZE, SEED_SIZE, cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < count; ++i) {
        cudaFree(d_mnemonic_data[i]);
        cudaFree(d_salt_data[i]);
    }
    cudaFree(d_mnemonics);
    cudaFree(d_salts);
    cudaFree(d_mnemonic_lens);
    cudaFree(d_salt_lens);
    cudaFree(d_out_seeds);

    std::vector<std::vector<unsigned char>> privs(count), chains(count);
    std::vector<std::string> keys(count, BITCOIN_SEED);
    std::vector<std::vector<unsigned char>> outputs;
    hmac_sha512_batch(keys, seeds, outputs, threads_per_block);  // 此处内部分也应有 kernel
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "[!] hmac_sha512_batch failed: " << cudaGetErrorString(err) << std::endl;
        CudaSafeCall(cudaDeviceSynchronize());
    }

    for (int i = 0; i < count; ++i) {
        privs[i] = std::vector<unsigned char>(outputs[i].begin(), outputs[i].begin() + 32);
        chains[i] = std::vector<unsigned char>(outputs[i].begin() + 32, outputs[i].end());
    }

    for (uint32_t index : path_indices) {
        std::vector<std::vector<unsigned char>> input_data(count);
        std::vector<uint8_t> hardened_u8(count);
        for (int i = 0; i < count; ++i) {
            bool hardened = index >= 0x80000000;
            hardened_u8[i] = hardened;
            input_data[i] = hardened ? privs[i] : derive_pubkey(privs[i], true);
        }

        std::vector<BYTE*> d_input_data(count);
        BYTE** d_inputs;
        uint8_t* d_hardened;
        uint32_t* d_indices;
        BYTE* d_datas;

        for (int i = 0; i < count; ++i) {
            CudaSafeCall(cudaMalloc(&d_input_data[i], input_data[i].size()));
            CudaSafeCall(cudaMemcpy(d_input_data[i], input_data[i].data(), input_data[i].size(), cudaMemcpyHostToDevice));
        }

        CudaSafeCall(cudaMalloc(&d_inputs, count * sizeof(BYTE*)));
        CudaSafeCall(cudaMemcpy(d_inputs, d_input_data.data(), count * sizeof(BYTE*), cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMalloc(&d_hardened, count * sizeof(uint8_t)));
        CudaSafeCall(cudaMalloc(&d_indices, count * sizeof(uint32_t)));
        std::vector<uint32_t> indices(count, index);
        CudaSafeCall(cudaMemcpy(d_hardened, hardened_u8.data(), count * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(d_indices, indices.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMalloc(&d_datas, count * MAX_CKD_DATA_SIZE));
        blocks = (count + threads_per_block - 1) / threads_per_block;
        ckd_data_kernel_batch<<<blocks, threads_per_block>>>(
            d_inputs, d_hardened, d_indices, d_datas, count);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                std::cerr << "[!] ckd_data_kernel_batch launch failed: " << cudaGetErrorString(err) << std::endl;
            CudaSafeCall(cudaDeviceSynchronize());
        }

        std::vector<std::vector<unsigned char>> prepared_data(count);
        for (int i = 0; i < count; ++i) {
            prepared_data[i].resize(hardened_u8[i] ? 1 + 32 + 4 : 33 + 4);
            CudaSafeCall(cudaMemcpy(
                prepared_data[i].data(),
                d_datas + i * MAX_CKD_DATA_SIZE,
                prepared_data[i].size(),
                cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < count; ++i)
            cudaFree(d_input_data[i]);
        cudaFree(d_inputs);
        cudaFree(d_hardened);
        cudaFree(d_indices);
        cudaFree(d_datas);

        std::vector<std::string> key_strs(count);
        for (int i = 0; i < count; ++i)
            key_strs[i] = std::string(reinterpret_cast<const char*>(chains[i].data()), chains[i].size());

        std::vector<std::vector<unsigned char>> out_hmac;
        hmac_sha512_batch(key_strs, prepared_data, out_hmac, threads_per_block);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                std::cerr << "[!] hmac_sha512_batch (CKD) failed: " << cudaGetErrorString(err) << std::endl;
            CudaSafeCall(cudaDeviceSynchronize());
        }

        for (int i = 0; i < count; ++i) {
            std::vector<unsigned char> IL(out_hmac[i].begin(), out_hmac[i].begin() + 32);
            std::vector<unsigned char> IR(out_hmac[i].begin() + 32, out_hmac[i].end());

            secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
            if (!secp256k1_ec_seckey_verify(ctx, IL.data())) {
                privs[i].clear();  // 无效 IL
                secp256k1_context_destroy(ctx);
                continue;
            }
            secp256k1_context_destroy(ctx);

            privs[i] = add_privkeys_mod_n(privs[i], IL);
            chains[i] = IR;
        }
    }

    final_pubkeys.clear();
    for (int i = 0; i < count; ++i) {
        if (!privs[i].empty())
            final_pubkeys.emplace_back(derive_pubkey(privs[i], false));  // 非压缩公钥
    }
}


// 包装 derive2pub 为 Python 接口
std::vector<std::string> py_derive_pubkeys(
    const std::vector<std::string>& mnemonics,
    const std::string& passphrase,
    const std::vector<uint32_t>& path_indices,
    int threads_per_block = 256
) {
    std::vector<std::vector<unsigned char>> pubkeys;
    derive2pub(mnemonics, passphrase, path_indices, pubkeys, threads_per_block);

    std::vector<std::string> hex_pubkeys;
    for (const auto& pub : pubkeys) {
        std::string hex;
        hex.reserve(pub.size() * 2);
        static const char* hexmap = "0123456789abcdef";
        for (unsigned char b : pub) {
            hex.push_back(hexmap[b >> 4]);
            hex.push_back(hexmap[b & 0x0F]);
        }
        hex_pubkeys.push_back(hex);
    }
    return hex_pubkeys;
}

PYBIND11_MODULE(pybind_derive2pub, m) {
    m.doc() = "GPU-accelerated BIP32 derive2pub binding";

    m.def("derive_pubkeys", &py_derive_pubkeys,
          py::arg("mnemonics"),
          py::arg("passphrase"),
          py::arg("path_indices"),
          py::arg("threads_per_block") = 256,
          R"pbdoc(
              Derive public keys from a list of mnemonics using GPU-accelerated BIP32.
              Returns a list of hex-encoded uncompressed public keys.
          )pbdoc");
}
