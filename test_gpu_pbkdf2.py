import gpu_pbkdf2

mnemonic = "aware report movie exile buyer drum poverty supreme gym oppose float elegant"
seed = gpu_pbkdf2.mnemonic_to_seed(mnemonic, "")
print("Seed:", seed.hex())

mnemonics = [
    "aware report movie exile buyer drum poverty supreme gym oppose float elegant",
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
]

seeds = gpu_pbkdf2.mnemonic_to_seed_batch(mnemonics, "")
for i, s in enumerate(seeds):
    print(f"Seed {i+1}:", s.hex())
