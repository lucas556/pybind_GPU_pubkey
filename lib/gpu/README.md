
## 编译PBKDF2
nvcc -O3 -Xcompiler -fPIC -std=c++17 -shared GPUPBKDF2.cu -o libgpupbkdf2.so     -gencode arch=compute_75,code=sm_75     -gencode arch=compute_86,code=sm_86     -gencode arch=compute_89,code=sm_89     -gencode arch=compute_90,code=sm_90
