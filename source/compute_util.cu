#include <cublas.h>
#include <cublas_v2.h>
#include "helper_functions.cuh"

__global__ void debug_ker(float* ptr, int addr){
    //int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("%d %f\n", addr, ptr[addr]);
}

void debug_array(float* arr, int N){
    for (int i = 0; i < N; ++i){
        debug_ker<<<1,1>>>(arr, i);
    }
    cudaDeviceSynchronize();
}

void row_major_sgemm(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float* C, float* tmp){
    float alpha = 1.0;
    float beta = 0.0;
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, tmp, m));
    checkCublasErrors(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, tmp, m, &beta, C, n, C, n));
}

void row_major_sgemm_add(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float*D, float* C, float* tmp){
    float alpha = 1.0;
    float beta = 0.0;
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, tmp, m));
    beta = 1.0;
    checkCublasErrors(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, tmp, m, &beta, D, n, C, n));
}
