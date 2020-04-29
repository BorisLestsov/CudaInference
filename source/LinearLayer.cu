#include <stdexcept>
#include "LinearLayer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"


LinearLayer::LinearLayer(cublasHandle_t& cublas_handle, const std::string& w_path):
    cublas_handle(cublas_handle)
{
    batch_size = 2;
    input_dim = 5*5*3;
    output_dim = 5;

    _w = new Tensor<float>({output_dim, input_dim});
    thrust::device_ptr<float> thr_ptr = thrust::device_pointer_cast<float>(_w->_ptr);
    thrust::fill(thr_ptr, thr_ptr + _w->count(), 0.5f);

    _tmp = new Tensor<float>({output_dim, batch_size});
    _res = new Tensor<float>({batch_size, output_dim});
}

LinearLayer::~LinearLayer(){
    delete _w; 
    delete _res; 
    delete _tmp;
}

void LinearLayer::forward() 
{
    //debug_ker<<<1,1>>>(_input->_ptr, 88);
    row_major_sgemm(cublas_handle, _input->size()[0], output_dim, input_dim, _input->_ptr, _w->_ptr, _res->_ptr, _tmp->_ptr);
}


void LinearLayer::set_input(Tensor<float>* input)
{
    if (input->size().size() != 2) {
        throw std::runtime_error("not two dims in input");
    }
    if (input->size()[1] != input_dim) {
        throw std::runtime_error("input dim is different");
    }
    _input = input;
}

Tensor<float>* LinearLayer::get_output()
{
    return _res;
}
