#include <stdexcept>
#include "ReluLayer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"
#include "npy.hpp"


ReluLayer::ReluLayer(int batch_size_p)
{
    batch_size = batch_size_p;
}

ReluLayer::~ReluLayer(){
    delete _res; 
}

class elwise_max_functor {
    public:
        elwise_max_functor() {}
        __host__ __device__ float operator()(float x) const 
        {
            return fmaxf(x, 0.0);
        }
};

__global__ void relu_ker(float* src, float* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = fmaxf(0.0, src[i]);
}

void ReluLayer::forward() 
{
    // thrust::device_ptr<float> thr_ptr = thrust::device_pointer_cast<float>(_input->_ptr);
    // thrust::device_ptr<float> thr_ptr2 = thrust::device_pointer_cast<float>(_res->_ptr);
    // thrust::transform(thr_ptr, thr_ptr + _input->count(), thr_ptr2, elwise_max_functor());

    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;
    int N = _input->count();

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    relu_ker<<<grid_size, block_size>>>(_input->_ptr, _res->_ptr, N);
}


void ReluLayer::set_input(Tensor<float>* input)
{
    if (input->size()[0] != batch_size) {
        throw std::runtime_error("batch size does not match");
    }
    _input = input;
    _res = new Tensor<float>(_input->size());
}

Tensor<float>* ReluLayer::get_output()
{
    return _res;
}
