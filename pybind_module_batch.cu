#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/GPU/GPUPBKDF2.cuh"
#include "include/GPU/GPUSHA512.cuh"

namespace py = pybind11;

// 单个调用
std::vector<uint8_t> mnemonic_to_seed(const std::string& mnemonic, const std::string& passphrase) {
    initSHA512Constants();
    return mnemonicToSeedGPU(mnemonic, passphrase);
}

// 批量调用（多助记词）
std::vector<std::vector<uint8_t>> mnemonic_to_seed_batch(const std::vector<std::string>& mnemonics, const std::string& passphrase) {
    initSHA512Constants();
    std::vector<std::vector<uint8_t>> results;
    results.reserve(mnemonics.size());
    for (const auto& m : mnemonics) {
        results.push_back(mnemonicToSeedGPU(m, passphrase));
    }
    return results;
}

PYBIND11_MODULE(gpu_pbkdf2, m) {
    m.doc() = "GPU PBKDF2-HMAC-SHA512 module with batch support";
    m.def("mnemonic_to_seed", &mnemonic_to_seed, "Single mnemonic to seed");
    m.def("mnemonic_to_seed_batch", &mnemonic_to_seed_batch, "Batch mnemonic to seeds");
}
