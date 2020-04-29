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


ReluLayer::ReluLayer(int el_size_p, int batch_size_p)
{
    batch_size = batch_size_p;
    el_size = el_size_p;

    _res = new Tensor<float>({batch_size, el_size});
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

void ReluLayer::forward() 
{

    thrust::device_ptr<float> thr_ptr = thrust::device_pointer_cast<float>(_input->_ptr);
    thrust::device_ptr<float> thr_ptr2 = thrust::device_pointer_cast<float>(_res->_ptr);
    thrust::transform(thr_ptr, thr_ptr + batch_size*el_size, thr_ptr2, elwise_max_functor());
}


void ReluLayer::set_input(Tensor<float>* input)
{
    if (input->size()[0] != batch_size) {
        throw std::runtime_error("batch size does not match");
    }
    if (input->count() / batch_size != el_size) {
        throw std::runtime_error(std::string("el size dont match") + std::to_string(input->count() / batch_size) + " vs " + std::to_string(el_size));
    }
    _input = input;
}

Tensor<float>* ReluLayer::get_output()
{
    return _res;
}
