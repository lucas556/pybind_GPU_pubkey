#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <secp256k1.h>

#include "GPUWrapper.cuh"

// ==== 工具函数 ====
__host__ void print_hex(const std::string& label, const std::vector<unsigned char>& data) {
    std::cout << label;
    for (uint8_t b : data) printf("%02x", b);
    std::cout << std::endl;
}

// ==== 公钥推导（可压缩或未压缩） ====
__host__ std::vector<unsigned char> derive_pubkey(const std::vector<unsigned char>& privkey, bool compressed = false) {
    if (privkey.size() != 32)
        throw std::runtime_error("Invalid private key length");

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx)
        throw std::runtime_error("Failed to create secp256k1 context");

    if (!secp256k1_ec_seckey_verify(ctx, privkey.data())) {
        secp256k1_context_destroy(ctx);
        throw std::runtime_error("Invalid secp256k1 private key");
    }

    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privkey.data())) {
        secp256k1_context_destroy(ctx);
        throw std::runtime_error("secp256k1_ec_pubkey_create failed");
    }

    std::vector<unsigned char> output(compressed ? 33 : 65);
    size_t len = output.size();
    int flag = compressed ? SECP256K1_EC_COMPRESSED : SECP256K1_EC_UNCOMPRESSED;
    if (!secp256k1_ec_pubkey_serialize(ctx, output.data(), &len, &pubkey, flag)) {
        secp256k1_context_destroy(ctx);
        throw std::runtime_error("secp256k1_ec_pubkey_serialize failed");
    }

    secp256k1_context_destroy(ctx);
    return output;
}

// ==== 私钥加法 mod n ====
__host__ std::vector<unsigned char> add_privkeys_mod_n(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    if (a.size() != 32 || b.size() != 32)
        throw std::runtime_error("Invalid private key or tweak length");

    std::vector<unsigned char> out = a;
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) throw std::runtime_error("Failed to create secp256k1 context");

    if (!secp256k1_ec_seckey_tweak_add(ctx, out.data(), b.data())) {
        secp256k1_context_destroy(ctx);
        throw std::runtime_error("secp256k1_ec_seckey_tweak_add failed");
    }

    secp256k1_context_destroy(ctx);
    return out;
}

// ==== BIP32 CKDpriv ====
__host__ std::pair<std::vector<unsigned char>, std::vector<unsigned char>> CKDpriv(
    const std::vector<unsigned char>& parent_privkey,
    const std::vector<unsigned char>& parent_chaincode,
    uint32_t index
) {
    bool hardened = index >= 0x80000000;
    std::vector<unsigned char> pubkey;
    if (!hardened)
        pubkey = derive_pubkey(parent_privkey, true);

    const std::vector<unsigned char>& left_or_pubkey = hardened ? parent_privkey : pubkey;

    std::vector<unsigned char> I = hmac_sha512_data(parent_chaincode, left_or_pubkey, hardened, index);
    std::vector<unsigned char> IL(I.begin(), I.begin() + 32);
    std::vector<unsigned char> IR(I.begin() + 32, I.end());

    std::vector<unsigned char> child_privkey = add_privkeys_mod_n(parent_privkey, IL);
    return {child_privkey, IR};
}

// ==== BIP32 路径解析 ====
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

// ==== 主程序入口 ====
int main() {
    std::string mnemonic = "aware report movie exile buyer drum poverty supreme gym oppose float elegant";
    std::string passphrase = "";
    std::string path = "m/44'/195'/0'/0/0";

    try {
        auto [master_privkey, master_chaincode] = derive_master_key(mnemonic, passphrase);
        print_hex("Master Private Key: ", master_privkey);
        print_hex("Master Chain Code : ", master_chaincode);

        std::vector<unsigned char> priv = master_privkey;
        std::vector<unsigned char> chain = master_chaincode;

        for (auto index : parse_path(path)) {
            auto [child_priv, child_chain] = CKDpriv(priv, chain, index);
            priv = child_priv;
            chain = child_chain;
            print_hex("[*] Derived Private Key: ", priv);
        }

        std::vector<unsigned char> pubkey = derive_pubkey(priv, false);  // false = uncompressed
        print_hex("[✓] Final Public Key : ", pubkey);

    } catch (const std::exception& e) {
        std::cerr << "[!] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
