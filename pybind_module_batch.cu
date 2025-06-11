#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include "include/GPU/GPUPBKDF2.cuh"
#include "include/GPU/GPUHMACWrapper.cuh"
#include "include/GPU/gpu_secp256k1/secp256k1.cuh"
#include "include/GPU/gpu_secp256k1/gpu_secp256k1_util.cuh"

namespace py = pybind11;
using ByteVec = std::vector<uint8_t>;
constexpr uint32_t HARDENED_OFFSET = 0x80000000;
const char* BITCOIN_SEED = "Bitcoin seed";

const ByteVec SECP256K1_N = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

void print_hex(const std::string& label, const ByteVec& data) {
    std::cout << label << ": ";
    for (uint8_t byte : data) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    std::cout << std::dec << std::endl;
}

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

ByteVec add_mod_order(const ByteVec& a, const ByteVec& b, const ByteVec& n) {
    ByteVec result(32);
    int carry = 0;
    for (int i = 31; i >= 0; --i) {
        int sum = a[i] + b[i] + carry;
        result[i] = sum & 0xFF;
        carry = sum >> 8;
    }
    bool greater_equal = true;
    for (int i = 0; i < 32; ++i) {
        if (result[i] < n[i]) { greater_equal = false; break; }
        if (result[i] > n[i]) break;
    }
    if (greater_equal) {
        int borrow = 0;
        for (int i = 31; i >= 0; --i) {
            int diff = result[i] - n[i] - borrow;
            if (diff < 0) {
                result[i] = diff + 256;
                borrow = 1;
            } else {
                result[i] = diff;
                borrow = 0;
            }
        }
    }
    return result;
}

extern ByteVec derive_unpublickey(const ByteVec& privkey);

py::dict derive_wallet_info(const std::string& mnemonic, const std::string& passphrase, const std::string& path) {
    initSHA512Constants();
    ByteVec seed = mnemonicToSeedGPU(mnemonic, passphrase);
    print_hex("[DEBUG] Seed", seed);

    ByteVec key(BITCOIN_SEED, BITCOIN_SEED + 12);
    ByteVec I = hmac_sha512_gpu(key, seed);
    if (I.size() != 64) {
        throw std::runtime_error("[ERROR] HMAC result not 64 bytes");
    }

    ByteVec privkey(32), chaincode(32);
    std::copy(I.begin(), I.begin() + 32, privkey.begin());
    std::copy(I.begin() + 32, I.end(), chaincode.begin());

    print_hex("[DEBUG] Master Private Key", privkey);
    print_hex("[DEBUG] Master Chain Code", chaincode);

    auto path_indices = parse_bip32_path(path);
    for (size_t i = 0; i < path_indices.size(); ++i) {
        uint32_t index = path_indices[i];

        if (privkey.size() != 32) {
            std::ostringstream oss;
            oss << "[ERROR] privkey length invalid: " << privkey.size();
            throw std::runtime_error(oss.str());
        }

        ByteVec data;
        data.push_back(0x00);
        data.insert(data.end(), privkey.begin(), privkey.end());
        data.push_back((index >> 24) & 0xFF);
        data.push_back((index >> 16) & 0xFF);
        data.push_back((index >> 8) & 0xFF);
        data.push_back(index & 0xFF);

        if (data.size() != 37) {
            std::ostringstream oss;
            oss << "[ERROR] Data length incorrect at level " << i << ": " << data.size();
            throw std::runtime_error(oss.str());
        }

        ByteVec I2 = hmac_sha512_gpu(chaincode, data);
        ByteVec IL(I2.begin(), I2.begin() + 32);
        ByteVec IR(I2.begin() + 32, I2.end());

        std::ostringstream oss;
        oss << "[DEBUG] IL (level " << i << ")";
        print_hex(oss.str(), IL);

        privkey = add_mod_order(IL, privkey, SECP256K1_N);

        if (std::all_of(privkey.begin(), privkey.end(), [](uint8_t b){ return b == 0; }))
            throw std::runtime_error("Invalid derived privkey: zero");

        std::ostringstream oss2;
        oss2 << "[DEBUG] privkey (level " << i << ")";
        print_hex(oss2.str(), privkey);

        std::ostringstream oss3;
        oss3 << "[DEBUG] chaincode (level " << i << ")";
        print_hex(oss3.str(), IR);

        chaincode = IR;
    }

    print_hex("[DEBUG] Final Private Key", privkey);

    ByteVec pubkey = derive_unpublickey(privkey);

    py::dict result;
    result["master_private_key"] = py::bytes(reinterpret_cast<const char*>(I.data()), 32);
    result["master_chain_code"] = py::bytes(reinterpret_cast<const char*>(I.data() + 32), 32);
    result["final_public_key"] = pubkey;
    result["final_private_key"] = privkey;
    result["final_chain_code"] = chaincode;
    result["seed"] = seed;
    return result;
}

PYBIND11_MODULE(pybind_GPU_pubkey, m) {
    m.doc() = "GPU-accelerated BIP32 wallet derivation";
    m.def("derive_wallet_info", &derive_wallet_info, "Derive public key from mnemonic and path");
}
