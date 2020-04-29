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


LinearLayer::LinearLayer(cublasHandle_t& cublas_handle, const std::string& w_path, int batch_size_p):
    cublas_handle(cublas_handle)
{
    batch_size = batch_size_p;

    std::vector<unsigned long> shape;
    std::vector<float> data;
    bool is_f;

    npy::LoadArrayFromNumpy(w_path + ".weight.npy", shape, is_f, data);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }
    output_dim = shape[0];
    input_dim = shape[1];

    _w = new Tensor<float>({output_dim, input_dim});
    _w->from_cpu(data.data());

    npy::LoadArrayFromNumpy(w_path + ".bias.npy", shape, is_f, data);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }
    _b = new Tensor<float>({batch_size, output_dim});
    float* tmp_ptr = _b->_ptr;
    for (int i = 0; i < batch_size; ++i){
        cudaMemcpy(tmp_ptr, data.data(), output_dim*sizeof(float), cudaMemcpyHostToDevice);
        tmp_ptr += output_dim;
    }

    _tmp = new Tensor<float>({output_dim, batch_size});
    _res = new Tensor<float>({batch_size, output_dim});
}

LinearLayer::~LinearLayer(){
    delete _w; 
    delete _b; 
    delete _res; 
    delete _tmp;
}

void LinearLayer::forward() 
{
    
    //row_major_sgemm(cublas_handle, batch_size, output_dim, input_dim, _input->_ptr, _w->_ptr, _res->_ptr, _tmp->_ptr);
    row_major_sgemm_add(cublas_handle, batch_size, output_dim, input_dim, _input->_ptr, _w->_ptr, _b->_ptr, _res->_ptr, _tmp->_ptr);
    //cudaDeviceSynchronize();
    //debug_ker<<<1,1>>>(_res->_ptr, 0);
}


void LinearLayer::set_input(Tensor<float>* input)
{
    if (input->size().size() != 2) {
        throw std::runtime_error("not two dims in input");
    }
    if (input->size()[0] != batch_size) {
        throw std::runtime_error("batch size does not match");
    }
    if (input->size()[1] != input_dim) {
        throw std::runtime_error(std::string("input dim is different: ") + std::to_string(input->size()[1]) + " vs " + std::to_string(input_dim));
    }
    _input = input;
}

Tensor<float>* LinearLayer::get_output()
{
    return _res;
}

int LinearLayer::get_output_dim()
{
    return output_dim;
}
