#pragma once
#include <vector>
#include <string>

// GPU PBKDF2-HMAC-SHA512 主入口
std::vector<unsigned char> mnemonicToSeedGPU(const std::string& mnemonic, const std::string& passphrase);
