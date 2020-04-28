#include "LinearLayer.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"


LinearLayer::LinearLayer()
{
    int size;
    size = 5*4;
    cudaMalloc(&_w, size*sizeof(float));
    thrust::device_ptr<float> thr_ptr = thrust::device_pointer_cast<float>(_w);
    thrust::fill(thr_ptr, thr_ptr + size, 0.5f);

    size = 3*4;
    cudaMalloc(&_res, size*sizeof(float));
    thrust::device_ptr<float> thr_ptr2 = thrust::device_pointer_cast<float>(_res);
    thrust::fill(thr_ptr2, thr_ptr2 + size, 0.0f);

    size = 3*4;
    cudaMalloc(&_tmp, size*sizeof(float));
    thrust::device_ptr<float> thr_ptr3 = thrust::device_pointer_cast<float>(_tmp);
    thrust::fill(thr_ptr3, thr_ptr3 + size, 0.0f);
}

void LinearLayer::forward() {}

void LinearLayer::forward_tmp(cublasHandle_t& cublas_handle, Tensor<float>* input)
{

    row_major_sgemm(cublas_handle, 3, 4, 5, input->_ptr, _w, _res, _tmp);
    cudaDeviceSynchronize();

    for (int i = 0; i < 3*4; ++i)
        debug_ker<<<1,1>>>(_res, i);
        cudaDeviceSynchronize();

}
