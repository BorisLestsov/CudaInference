#ifndef CUDA_PROJ_COMPUTE_UTIL_CUH
#define CUDA_PROJ_COMPUTE_UTIL_CUH

#include <cublas.h>
#include <cublas_v2.h>

__global__ void debug_ker(float* ptr, int addr);
void debug_array(float* arr, int N);

void row_major_sgemm(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float* C, float* tmp);
void row_major_sgemm_add(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float*D, float* C, float* tmp);

template<typename T>
void cuda_add(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_sub(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_mul(T* src1, T* src2, T* res, int N);
template<typename T>
void cuda_div(T* src1, T* src2, T* res, int N);

template<typename T>
__global__ void transpose_ker(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims);
template<typename T>
void cuda_transpose(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims, int N);

#endif //CUDA_PROJ_TENSOR_CUH
