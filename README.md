## mnemonic to pubkey

```
git clone https://github.com/lucas556/pybind_GPU_pubkey.git
cd pybind_GPU_pubkey/src
nvcc -Xcompiler -fPIC -shared -std=c++17   -gencode arch=compute_75,code=sm_75   -gencode arch=compute_86,code=sm_86   -gencode arch=compute_89,code=sm_89   -gencode arch=compute_90,code=sm_90   pybind_mnemonic2pub.cu -o pybind_derive2pub.so   -I/usr/include/python3.10 -I/usr/include/pybind11   -lsecp256k1
```

### maxrregcount test
```
nvcc -Xcompiler -fPIC -shared -std=c++17   -O3 -Xptxas -v -maxrregcount=32   -gencode arch=compute_75,code=sm_75   -gencode arch=compute_86,code=sm_86   -gencode arch=compute_89,code=sm_89   -gencode arch=compute_90,code=sm_90   pybind_mnemonic2pub.cu -o pybind_derive2pub.so   -I/usr/include/python3.10 -I/usr/include/pybind11   -lsecp256k1
```
### python 中使用
```
import pybind_derive2pub
mnemo = Mnemonic("english")
mnemonics = [mnemo.generate(strength=128) for _ in range(200000)]

# 派生路径 m/44'/0'/0'/0/0（可自定义）
path_indices = [44 | 0x80000000, 0 | 0x80000000, 0 | 0x80000000, 0, 0]
passphrase = ""

# 执行 GPU 加速派生
pubkeys = pybind_derive2pub.derive_pubkeys(
    mnemonics=mnemonics,
    passphrase=passphrase,
    path_indices=path_indices,
    threads_per_block=128
)

# 输出所有公钥结果
for i, pk in enumerate(pubkeys):
    print(f"[{i}] {pk}")
```

### build test_mnemonic2Seed.cu

```
nvcc -O3 -Xcompiler -fPIC -std=c++17 test_mnemonic2Seed.cu -o test_mnemonic2Seed -I. -lcudart -lsecp256k1 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90
```
