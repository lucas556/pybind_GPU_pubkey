#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include "include/GPU/GPUPBKDF2.cuh"
#include "include/GPU/GPUHMACWrapper.cuh"
#include "include/GPU/GPUSecpKernel.h"

namespace py = pybind11;
using ByteVec = std::vector<uint8_t>;
constexpr uint32_t HARDENED_OFFSET = 0x80000000;
const char* BITCOIN_SEED = "Bitcoin seed";

// secp256k1 order
const ByteVec SECP256K1_N = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

std::vector<uint32_t> parse_bip32_path(const std::string& path) {
    std::vector<uint32_t> result;
    std::stringstream ss(path);
    std::string item;
    while (std::getline(ss, item, '/')) {
        if (item == "m" || item.empty()) continue;
        uint32_t hardened = 0;
        if (item.back() == '\'') {
            hardened = HARDENED_OFFSET;
            item.pop_back();
        }
        result.push_back(std::stoul(item) | hardened);
    }
    return result;
}

// 字节序加法，返回: (a + b) % n
ByteVec add_mod_n(const ByteVec& a, const ByteVec& b, const ByteVec& n) {
    ByteVec result(32);
    int carry = 0;
    for (int i = 31; i >= 0; --i) {
        int sum = a[i] + b[i] + carry;
        result[i] = sum & 0xFF;
        carry = sum >> 8;
    }
    // 如果结果 >= n，则减去 n
    bool ge = false;
    for (int i = 0; i < 32; ++i) {
        if (result[i] > n[i]) { ge = true; break; }
        if (result[i] < n[i]) break;
    }
    if (ge) {
        carry = 0;
        for (int i = 31; i >= 0; --i) {
            int diff = result[i] - n[i] - carry;
            if (diff < 0) {
                result[i] = diff + 256;
                carry = 1;
            } else {
                result[i] = diff;
                carry = 0;
            }
        }
    }
    return result;
}

py::dict derive_wallet_info(const std::string& mnemonic, const std::string& passphrase, const std::string& path) {
    ByteVec seed = mnemonicToSeedGPU(mnemonic, passphrase);
    ByteVec key_str(BITCOIN_SEED, BITCOIN_SEED + strlen(BITCOIN_SEED));
    ByteVec I = run_hmac_sha512_gpu(key_str, seed);
    ByteVec privkey(I.begin(), I.begin() + 32);
    ByteVec chaincode(I.begin() + 32, I.end());

    for (uint32_t index : parse_bip32_path(path)) {
        ByteVec data(1 + 32 + 4);
        data[0] = 0x00;
        std::copy(privkey.begin(), privkey.end(), data.begin() + 1);
        data[33] = (index >> 24) & 0xFF;
        data[34] = (index >> 16) & 0xFF;
        data[35] = (index >> 8) & 0xFF;
        data[36] = index & 0xFF;

        ByteVec hmac = run_hmac_sha512_gpu(chaincode, data);
        ByteVec IL(hmac.begin(), hmac.begin() + 32);
        chaincode.assign(hmac.begin() + 32, hmac.end());

        privkey = add_mod_n(privkey, IL, SECP256K1_N);
    }

    ByteVec pubkey = derive_public_key_from_private(privkey);

    py::dict result;
    result["seed"] = py::bytes(reinterpret_cast<const char*>(seed.data()), seed.size());
    result["master_private_key"] = py::bytes(reinterpret_cast<const char*>(I.data()), 32);
    result["chain_code"] = py::bytes(reinterpret_cast<const char*>(chaincode.data()), 32);
    result["final_private_key"] = py::bytes(reinterpret_cast<const char*>(privkey.data()), 32);
    result["final_public_key"] = py::bytes(reinterpret_cast<const char*>(pubkey.data()), pubkey.size());

    return result;
}

PYBIND11_MODULE(pygpupub, m) {
    m.doc() = "GPU-based full BIP32 path derivation";
    m.def("derive_wallet_info", &derive_wallet_info, "Derive final priv/pubkey from mnemonic and BIP32 path");
}
