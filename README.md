## mnemonic to pubkey

```
git clone
cd
nvcc -Xcompiler -fPIC -shared -std=c++17   -gencode arch=compute_75,code=sm_75   -gencode arch=compute_86,code=sm_86   -gencode arch=compute_89,code=sm_89   -gencode arch=compute_90,code=sm_90   pybind_mnemonic2pub.cu -o pybind_derive2pub.so   -I/usr/include/python3.10 -I/usr/include/pybind11   -lsecp256k1
```
