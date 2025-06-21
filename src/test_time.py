import time
import random
import string
from mnemonic import Mnemonic

import pybind_derive2pub

mnemo = Mnemonic("english")
mnemonics = [mnemo.generate(strength=128) for _ in range(100000)]

# 派生路径 m/44'/0'/0'/0/0（可自定义）
path_indices = [44 | 0x80000000, 0 | 0x80000000, 0 | 0x80000000, 0, 0]
passphrase = ""

start_time = time.time()

pubkeys = pybind_derive2pub.derive_pubkeys(
    mnemonics=mnemonics,
    passphrase=passphrase,
    path_indices=path_indices,
    threads_per_block=256
)

for i, pk in enumerate(pubkeys):
    print(f"[{i}] {pk}")

# 输出耗时
elapsed = time.time() - start_time
print(f"\n[✓] Total {len(pubkeys)} pubkeys derived in {elapsed:.2f} seconds")
