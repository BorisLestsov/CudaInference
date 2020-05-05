#include <stdexcept>
#include "LinearLayer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"
#include "npy.hpp"


LinearLayer::LinearLayer(cublasHandle_t& cublas_handle, const std::string& w_path, bool bias_p):
    cublas_handle(cublas_handle),
    _bias(bias_p)
{

    std::vector<unsigned long> shape;
    std::vector<float> data;
    bool is_f;

    npy::LoadArrayFromNumpy(w_path + ".weight.npy", shape, is_f, data);
    output_dim = shape[0];
    input_dim = shape[1];

    _w = std::shared_ptr<Tensor<float>>(new Tensor<float>({output_dim, input_dim}));
    _w->from_cpu(data.data());

    if (_bias){
        npy::LoadArrayFromNumpy(w_path + ".bias.npy", shape, is_f, data_b);
    }
}

LinearLayer::~LinearLayer(){
}

void LinearLayer::forward() 
{
    
    if (_bias){
        row_major_sgemm_add(cublas_handle, batch_size, output_dim, input_dim, _input->_ptr, _w->_ptr, _b->_ptr, _res->_ptr, _tmp->_ptr);
    } else {
        row_major_sgemm(cublas_handle, batch_size, output_dim, input_dim, _input->_ptr, _w->_ptr, _res->_ptr, _tmp->_ptr);
    }
}


void LinearLayer::set_input(std::shared_ptr<Tensor<float>> input)
{
    batch_size = input->size()[0];
    int inp_w = input->count() / batch_size;
    if (inp_w != input_dim) {
        throw std::runtime_error(std::string("input dim is different: ") + std::to_string(inp_w) + " vs " + std::to_string(input_dim));
    }

    if (_bias){
        _b = std::shared_ptr<Tensor<float>>(new Tensor<float>({batch_size, output_dim}));
        float* tmp_ptr = _b->_ptr;
        for (int i = 0; i < batch_size; ++i){
            cudaMemcpy(tmp_ptr, data_b.data(), output_dim*sizeof(float), cudaMemcpyHostToDevice);
            tmp_ptr += output_dim;
        }
    }

    _input = input;
    _tmp = std::shared_ptr<Tensor<float>>(new Tensor<float>({output_dim, batch_size}));
    _res = std::shared_ptr<Tensor<float>>(new Tensor<float>({batch_size, output_dim}));
}

std::shared_ptr<Tensor<float>> LinearLayer::get_output()
{
    return _res;
}

int LinearLayer::get_output_dim()
{
    return output_dim;
}
