#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <secp256k1.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"
#include "GPUHMAC.cuh"

namespace py = pybind11;

constexpr const char* BITCOIN_SEED = "Bitcoin seed";
using ByteVec = std::vector<uint8_t>;

__host__ std::vector<unsigned char> derive_pubkey(const std::vector<unsigned char>& privkey, secp256k1_context* ctx, bool compressed = true) {
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

    return output;
}

__host__ std::vector<unsigned char> add_privkeys_mod_n(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, secp256k1_context* ctx) {
    std::vector<unsigned char> out = a;
    if (!secp256k1_ec_seckey_tweak_add(ctx, out.data(), b.data()))
        throw std::runtime_error("secp256k1_ec_seckey_tweak_add failed");
    return out;
}

__host__ void gpu_pbkdf2_batch(const std::vector<std::string>& mnemonics, const std::string& passphrase, std::vector<ByteVec>& out_seeds, int threads_per_block) {
    int count = mnemonics.size();
    std::vector<std::string> salts(count, "mnemonic" + passphrase);

    std::vector<const char*> h_mnemonics(count), h_salts(count);
    std::vector<size_t> mnemonic_lens(count), salt_lens(count);
    for (int i = 0; i < count; ++i) {
        h_mnemonics[i] = mnemonics[i].c_str();
        h_salts[i] = salts[i].c_str();
        mnemonic_lens[i] = mnemonics[i].size();
        salt_lens[i] = salts[i].size();
    }

    char **d_mnemonics, **d_salts;
    size_t *d_mnemonic_lens, *d_salt_lens;
    BYTE *d_out_seeds;

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
    pbkdf2_kernel<<<blocks, threads_per_block>>>(d_mnemonics, d_mnemonic_lens, d_salts, d_salt_lens,
        PBKDF2_HMAC_SHA512_ITERATIONS, d_out_seeds, count);
    CudaSafeCall(cudaGetLastError());
    CudaSafeCall(cudaDeviceSynchronize());

    out_seeds.resize(count);
    for (int i = 0; i < count; ++i) {
        out_seeds[i].resize(SEED_SIZE);
        CudaSafeCall(cudaMemcpy(out_seeds[i].data(), d_out_seeds + i * SEED_SIZE, SEED_SIZE, cudaMemcpyDeviceToHost));
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
}

std::vector<std::vector<unsigned char>> py_derive_pubkeys(
    const std::vector<std::string>& mnemonics,
    const std::string& passphrase,
    const std::vector<uint32_t>& path_indices,
    int threads_per_block = 256
) {
    initSHA512Constants();
    std::vector<ByteVec> pbkdf2_seeds;
    gpu_pbkdf2_batch(mnemonics, passphrase, pbkdf2_seeds, threads_per_block);

    int count = mnemonics.size();
    std::vector<std::vector<unsigned char>> privs(count), chains(count);
    std::vector<std::string> master_key_labels(count, BITCOIN_SEED);
    std::vector<ByteVec> master_hmacs;
    hmac_sha512_batch(master_key_labels, pbkdf2_seeds, master_hmacs, threads_per_block);

    for (int i = 0; i < count; ++i) {
        privs[i].assign(master_hmacs[i].begin(), master_hmacs[i].begin() + 32);
        chains[i].assign(master_hmacs[i].begin() + 32, master_hmacs[i].end());
    }

    static secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) throw std::runtime_error("Failed to create secp256k1 context");

    for (uint32_t index : path_indices) {
        std::vector<ByteVec> input_data(count);
        std::vector<uint8_t> hardened(count);

        for (int i = 0; i < count; ++i) {
            if (privs[i].empty()) continue;
            hardened[i] = index >= 0x80000000;
            input_data[i] = hardened[i] ? privs[i] : derive_pubkey(privs[i], ctx, true);
        }

        std::vector<ByteVec> prepared(count);
        for (int i = 0; i < count; ++i) {
            if (privs[i].empty()) continue;
            if (hardened[i]) prepared[i].push_back(0x00);
            prepared[i].insert(prepared[i].end(), input_data[i].begin(), input_data[i].end());
            prepared[i].push_back((index >> 24) & 0xFF);
            prepared[i].push_back((index >> 16) & 0xFF);
            prepared[i].push_back((index >> 8) & 0xFF);
            prepared[i].push_back(index & 0xFF);
        }

        std::vector<std::string> key_strs(count);
        for (int i = 0; i < count; ++i)
            if (!privs[i].empty())
                key_strs[i].assign(chains[i].begin(), chains[i].end());

        std::vector<ByteVec> out_hmac;
        hmac_sha512_batch(key_strs, prepared, out_hmac, threads_per_block);

        for (int i = 0; i < count; ++i) {
            if (privs[i].empty()) continue;

            std::vector<unsigned char> IL(out_hmac[i].begin(), out_hmac[i].begin() + 32);
            std::vector<unsigned char> IR(out_hmac[i].begin() + 32, out_hmac[i].end());

            if (!secp256k1_ec_seckey_verify(ctx, IL.data())) {
                privs[i].clear();
                continue;
            }

            privs[i] = add_privkeys_mod_n(privs[i], IL, ctx);
            chains[i] = IR;
        }
    }

    std::vector<std::vector<unsigned char>> final_pubkeys;
    for (const auto& priv : privs) {
        if (!priv.empty()) {
            final_pubkeys.push_back(derive_pubkey(priv, ctx, false));
        }
    }

    return final_pubkeys;
}

PYBIND11_MODULE(pybind_derive2pub, m) {
    m.doc() = "Optimized GPU-accelerated BIP32 derive2pub";
    m.def("derive_pubkeys", &py_derive_pubkeys,
          py::arg("mnemonics"),
          py::arg("passphrase"),
          py::arg("path_indices"),
          py::arg("threads_per_block") = 256);
}
