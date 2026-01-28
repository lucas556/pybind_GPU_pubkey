

### 代码说明

#### 最新
```
最新:
1. 修正 GPUHMAC.cuh  GPUPBKDF2.cuh  GPUSHA512.cuh 之间的关系,增加批量batch
2. secp256k1使用多进程并发 解决GPU等待CPU的问题 多核并行/去掉 verify(IL) → ECC 调用次数减少一点/去掉 verify(IL) → ECC 调用次数/缓存 pub + pubkey_tweak_add 代替重复 k·G 
3. 当前速度90k/s(3080)
4. 出于安全考虑 不在公开最新版本
```

```
secp256k1 使用libsecp256k1 : https://github.com/bitcoin-core/secp256k1
助记词验证使用的hashlib.sha256

1. 可在MnemonicValidator选择助记词的数量
2. progress_checkpoint.txt储存的是batch_id,当batch_size_cpu发生变化或者修改,这里需要修改
3. threads_per_block = 256 这里使用的是3080的显卡,增大会爆内存
4. batch_size_cpu 单次cpu处理最大的助记词数量
5. sub_batch_size 每线程处理的数量
6. pool = Pool(cpu_count() * 2) 进程数量
7. samples_logged.txt 为随机选择的助记词和公钥 方便测试 可删除

```

### 使用方法
```
1. 修改target_pubkey_hex为需要的公钥,其中使用链上提取出的公钥前面加 '04'
2. 编辑词表组合 编辑已知的助记词
3. 运行程序
nohup python3 mnemonic2pub.py > run.log 2>&1 &
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
