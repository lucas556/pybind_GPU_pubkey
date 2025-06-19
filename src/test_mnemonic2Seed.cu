#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <secp256k1.h>

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"
#include "GPUHMAC.cuh"

#define BITCOIN_SEED "Bitcoin seed"
#define MAX_CKD_DATA_SIZE (1 + 32 + 4)

__host__ void print_hex(const std::string& label, const std::vector<unsigned char>& data) {
    std::cout << label;
    for (uint8_t b : data) printf("%02x", b);
    std::cout << std::endl;
}

__global__ void ckd_data_kernel(
    unsigned char** pubkeys_or_lefts,  // 注意：取消 const BYTE**
    const uint8_t* hardened,
    const uint32_t* indices,
    BYTE* datas,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int offset = idx * (33 + 4);
    const BYTE* input = pubkeys_or_lefts[idx];
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

__host__ std::vector<unsigned char> add_privkeys_mod_n(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    std::vector<unsigned char> out = a;
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!secp256k1_ec_seckey_tweak_add(ctx, out.data(), b.data()))
        throw std::runtime_error("secp256k1_ec_seckey_tweak_add failed");
    secp256k1_context_destroy(ctx);
    return out;
}

std::vector<uint32_t> parse_path(const std::string& path) {
    std::vector<uint32_t> result;
    if (path.empty() || path[0] != 'm') throw std::runtime_error("Invalid path");
    size_t i = 2;
    while (i < path.size()) {
        size_t slash = path.find('/', i);
        std::string token = path.substr(i, slash - i);
        if (!token.empty()) {
            if (token.back() == '\'')
                result.push_back(0x80000000 + std::stoi(token.substr(0, token.size() - 1)));
            else
                result.push_back(std::stoi(token));
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
    return result;
}

__host__ void derive_keys_from_mnemonics(
    const std::vector<std::string>& mnemonics,
    const std::string& passphrase,
    const std::string& path,
    std::vector<std::vector<unsigned char>>& final_privkeys,
    std::vector<std::vector<unsigned char>>& final_chaincodes,
    int threads_per_block = 256
) {
    int count = mnemonics.size();
    initSHA512Constants();

    std::vector<std::string> salts(count);
    for (int i = 0; i < count; ++i) salts[i] = "mnemonic" + passphrase;

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
    pbkdf2_kernel<<<blocks, threads_per_block>>>(d_mnemonics, d_mnemonic_lens, d_salts, d_salt_lens, PBKDF2_HMAC_SHA512_ITERATIONS, d_out_seeds, count);
    CudaSafeCall(cudaDeviceSynchronize());

    std::vector<std::vector<unsigned char>> privs(count), chains(count), seeds(count);
    for (int i = 0; i < count; ++i) {
        seeds[i].resize(SEED_SIZE);
        CudaSafeCall(cudaMemcpy(seeds[i].data(), d_out_seeds + i * SEED_SIZE, SEED_SIZE, cudaMemcpyDeviceToHost));
    }

    std::vector<std::string> keys(count, BITCOIN_SEED);
    std::vector<std::vector<unsigned char>> outputs;
    hmac_sha512_batch(keys, seeds, outputs, threads_per_block);
    for (int i = 0; i < count; ++i) {
        privs[i] = std::vector<unsigned char>(outputs[i].begin(), outputs[i].begin() + 32);
        chains[i] = std::vector<unsigned char>(outputs[i].begin() + 32, outputs[i].end());
    }

    for (int i = 0; i < count; ++i) {
        cudaFree(d_mnemonic_data[i]);
        cudaFree(d_salt_data[i]);
    }
    cudaFree(d_mnemonics); cudaFree(d_salts);
    cudaFree(d_mnemonic_lens); cudaFree(d_salt_lens);
    cudaFree(d_out_seeds);

    std::vector<uint32_t> path_indices = parse_path(path);
    for (uint32_t index : path_indices) {
        std::vector<std::vector<unsigned char>> input_data(count);
        std::vector<uint8_t> hardened_u8(count);
        for (int i = 0; i < count; ++i) {
            bool hardened = index >= 0x80000000;
            hardened_u8[i] = hardened;
            input_data[i] = hardened ? privs[i] : derive_pubkey(privs[i], true);
        }

        uint8_t* d_hardened;
        uint32_t* d_indices;
        BYTE* d_datas;
        CudaSafeCall(cudaMalloc(&d_hardened, count * sizeof(uint8_t)));
        CudaSafeCall(cudaMalloc(&d_indices, count * sizeof(uint32_t)));

        std::vector<uint32_t> indices(count, index);
        CudaSafeCall(cudaMemcpy(d_hardened, hardened_u8.data(), count * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(d_indices, indices.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));

        std::vector<BYTE*> d_input_data(count);
        for (int i = 0; i < count; ++i) {
            CudaSafeCall(cudaMalloc(&d_input_data[i], input_data[i].size()));
            CudaSafeCall(cudaMemcpy(d_input_data[i], input_data[i].data(), input_data[i].size(), cudaMemcpyHostToDevice));
        }
        BYTE** d_inputs;
        CudaSafeCall(cudaMalloc(&d_inputs, count * sizeof(BYTE*)));
        CudaSafeCall(cudaMemcpy(d_inputs, d_input_data.data(), count * sizeof(BYTE*), cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMalloc(&d_datas, count * MAX_CKD_DATA_SIZE));
        blocks = (count + threads_per_block - 1) / threads_per_block;
        ckd_data_kernel<<<blocks, threads_per_block>>>(d_inputs, d_hardened, d_indices, d_datas, count);
        CudaSafeCall(cudaDeviceSynchronize());

        std::vector<std::vector<unsigned char>> prepared_data(count);
        for (int i = 0; i < count; ++i) {
            prepared_data[i].resize(hardened_u8[i] ? 1 + 32 + 4 : 33 + 4);
            CudaSafeCall(cudaMemcpy(prepared_data[i].data(), d_datas + i * MAX_CKD_DATA_SIZE, prepared_data[i].size(), cudaMemcpyDeviceToHost));
        }

        std::vector<std::string> key_strs(count);
        for (int i = 0; i < count; ++i)
            key_strs[i] = std::string(reinterpret_cast<const char*>(chains[i].data()), chains[i].size());

        std::vector<std::vector<unsigned char>> out_hmac;
        hmac_sha512_batch(key_strs, prepared_data, out_hmac, threads_per_block);

        for (int i = 0; i < count; ++i) {
            std::vector<unsigned char> IL(out_hmac[i].begin(), out_hmac[i].begin() + 32);
            std::vector<unsigned char> IR(out_hmac[i].begin() + 32, out_hmac[i].end());
            privs[i] = add_privkeys_mod_n(privs[i], IL);
            chains[i] = IR;
        }

        for (int i = 0; i < count; ++i)
            cudaFree(d_input_data[i]);
        cudaFree(d_inputs);
        cudaFree(d_hardened);
        cudaFree(d_indices);
        cudaFree(d_datas);
    }

    final_privkeys = privs;
    final_chaincodes = chains;
}

int main() {
    std::string mnemonic = "aware report movie exile buyer drum poverty supreme gym oppose float elegant";
    std::string passphrase = "";
    std::string path = "m/44'/195'/0'/0/0";

    std::vector<std::vector<unsigned char>> privs, chains;
    try {
        derive_keys_from_mnemonics({mnemonic}, passphrase, path, privs, chains, 256);
        print_hex("[✓] Final Private Key : ", privs[0]);
        print_hex("[✓] Final Chain Code  : ", chains[0]);
        print_hex("[✓] Final Public Key  : ", derive_pubkey(privs[0], false));
    } catch (const std::exception& e) {
        std::cerr << "[!] Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
