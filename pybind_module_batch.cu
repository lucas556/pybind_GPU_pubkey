#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/GPU/GPUPBKDF2.cuh"
#include "include/GPU/GPUSHA512.cuh"

namespace py = pybind11;

// 单个助记词 → Python bytes 类型
py::bytes mnemonic_to_seed(const std::string& mnemonic, const std::string& passphrase) {
    initSHA512Constants();
    auto seed = mnemonicToSeedGPU(mnemonic, passphrase);
    return py::bytes(reinterpret_cast<const char*>(seed.data()), seed.size());
}

// 批量助记词 → Python bytes 列表
std::vector<py::bytes> mnemonic_to_seed_batch(const std::vector<std::string>& mnemonics, const std::string& passphrase) {
    initSHA512Constants();
    std::vector<py::bytes> results;
    results.reserve(mnemonics.size());
    for (const auto& m : mnemonics) {
        auto seed = mnemonicToSeedGPU(m, passphrase);
        results.emplace_back(reinterpret_cast<const char*>(seed.data()), seed.size());
    }
    return results;
}

PYBIND11_MODULE(gpu_pbkdf2, m) {
    m.doc() = "GPU PBKDF2-HMAC-SHA512 module with Python-native bytes return";
    m.def("mnemonic_to_seed", &mnemonic_to_seed, "Single mnemonic to seed (returns bytes)");
    m.def("mnemonic_to_seed_batch", &mnemonic_to_seed_batch, "Batch mnemonic to seed list (returns list of bytes)");
}
