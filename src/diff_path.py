import pybind_derive2pub
import binascii

mnemonic = "aware report movie exile buyer drum poverty supreme gym oppose float elegant"
passphrase = ""

# Bitcoin: m/44'/0'/0'/0/0
btc_path = [44 | 0x80000000, 0 | 0x80000000, 0 | 0x80000000, 0, 0]

# TRON: m/44'/195'/0'/0/0
tron_path = [44 | 0x80000000, 195 | 0x80000000, 0 | 0x80000000, 0, 0]

threads_per_block = 128

print("[INFO] Deriving Bitcoin path pubkey...")
btc_pubkey = pybind_derive2pub.derive_pubkeys(
    mnemonics=[mnemonic],
    passphrase=passphrase,
    path_indices=btc_path,
    threads_per_block=threads_per_block
)[0]

print("[INFO] Deriving Tron path pubkey...")
tron_pubkey = pybind_derive2pub.derive_pubkeys(
    mnemonics=[mnemonic],
    passphrase=passphrase,
    path_indices=tron_path,
    threads_per_block=threads_per_block
)[0]

print("\n[RESULT] BIP44 Bitcoin m/44'/0'/0'/0/0 公钥:")
print(bytes(btc_pubkey).hex())
print("\n[RESULT] BIP44 TRON m/44'/195'/0'/0/0 公钥:")
print(bytes(tron_pubkey).hex())
