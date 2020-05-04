#ifndef CUDA_PROJ_COMPUTE_UTIL_CUH
#define CUDA_PROJ_COMPUTE_UTIL_CUH

#include <cublas.h>
#include <cublas_v2.h>

__global__ void debug_ker(float* ptr, int addr);
void debug_array(float* arr, int N);

void row_major_sgemm(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float* C, float* tmp);
void row_major_sgemm_add(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float*D, float* C, float* tmp);

template<typename T>
void cuda_add_inplace(T* src1, T* src2, T* res, int N);

#endif //CUDA_PROJ_TENSOR_CUH
