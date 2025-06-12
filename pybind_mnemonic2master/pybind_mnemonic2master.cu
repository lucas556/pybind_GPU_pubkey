#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GPUPBKDF2.cuh"
#include "GPUHMACWrapper.cuh"
#include "GPUSHA512.cuh"

namespace py = pybind11;
using ByteVec = std::vector<uint8_t>;

const char* BITCOIN_SEED = "Bitcoin seed";

std::pair<ByteVec, ByteVec> mnemonic_to_master_key(const std::string& mnemonic, const std::string& passphrase) {
    ByteVec seed = mnemonicToSeedGPU(mnemonic, passphrase);
    ByteVec key(BITCOIN_SEED, BITCOIN_SEED + strlen(BITCOIN_SEED));
    ByteVec I = hmac_sha512_gpu(key, seed);

    if (I.size() != 64) {
        throw std::runtime_error("[ERROR] HMAC result not 64 bytes");
    }

    ByteVec master_privkey(I.begin(), I.begin() + 32);
    ByteVec master_chaincode(I.begin() + 32, I.end());

    return {master_privkey, master_chaincode};
}

PYBIND11_MODULE(pybind_mnemonic2master, m) {
    m.doc() = "Minimal GPU-accelerated mnemonic → master key module";

    initSHA512Constants();  // 初始化 SHA-512 K 常量

    m.def("mnemonic_to_master_key", [](const std::string& mnemonic, const std::string& passphrase) {
        auto [privkey, chaincode] = mnemonic_to_master_key(mnemonic, passphrase);

        py::bytes privkey_bytes(reinterpret_cast<const char*>(privkey.data()), privkey.size());
        py::bytes chaincode_bytes(reinterpret_cast<const char*>(chaincode.data()), chaincode.size());

        return py::make_tuple(privkey_bytes, chaincode_bytes);
    }, py::arg("mnemonic"), py::arg("passphrase") = "");
}
