#include <stdexcept>
#include <string>
#include "BatchNormLayer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"
#include "npy.hpp"



BatchNormLayer::BatchNormLayer(const std::string& w_path, int batch_size_p):
    batch_size(batch_size_p)
{
    std::vector<unsigned long> shape;
    std::vector<float> data;
    bool is_f;

    npy::LoadArrayFromNumpy(w_path + ".weight.npy", shape, is_f, data_w);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }
    npy::LoadArrayFromNumpy(w_path + ".bias.npy", shape, is_f, data_b);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }
    npy::LoadArrayFromNumpy(w_path + ".running_mean.npy", shape, is_f, data_rm);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }
    npy::LoadArrayFromNumpy(w_path + ".running_var.npy", shape, is_f, data_rv);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }
}

BatchNormLayer::~BatchNormLayer(){
    delete _res; 
}


__global__ 
void batchnorm2d(float* src, float* res, float* w, float* b, float* rm, float* rv, int C, int Ho, int Wo, int batch_size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int total_scals = C*Ho*Wo*batch_size;

    if (i >= total_scals){
        return;
    }
    
    //res[ni*o_mat_stride + ci*(Ho*Wo) + hi*(Wo) + wi] = ;

    res[i] = (src[i] - rm[i]) / (rv[i]) * w[i] + b[i];
}

void BatchNormLayer::forward() 
{
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    int total = batch_size*C*Ho*Wo;
    num_blocks_x = total/cell_size + (total % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    batchnorm2d<<<grid_size, block_size>>>(_input->_ptr, _res->_ptr, _w->_ptr, _b->_ptr, _rm->_ptr, _rv->_ptr, C, Ho, Wo, batch_size);
    //debug_array(_res->_ptr, _res->count());

}


void BatchNormLayer::set_input(Tensor<float>* input)
{
    if (input->size().size() != 4) {
        throw std::runtime_error("not four dims in input");
    }

    Size isize = input->size();
    //batch_size = isize[0];
    Hi = isize[2];
    Wi = isize[3];
    Ho = Hi;
    Wo = Wi;
    C = isize[1];


    if (input->size()[0] != batch_size) {
        throw std::runtime_error("batch size does not match");
    }
    _input = input;

    _res = new Tensor<float>({batch_size, C, Ho, Wo});

    _w = new Tensor<float>({batch_size, C, Ho, Wo});
    _b = new Tensor<float>({batch_size, C, Ho, Wo});
    _rm = new Tensor<float>({batch_size, C, Ho, Wo});
    _rv = new Tensor<float>({batch_size, C, Ho, Wo});

    thrust::device_ptr<float> thr_ptr;

    thr_ptr = thrust::device_pointer_cast<float>(_w->_ptr);
    for (int i = 0; i < batch_size; ++i){
        for (int j = 0; j < C; ++j){
            thrust::fill(thr_ptr, thr_ptr + Ho*Wo, data_w[j]);
            thr_ptr += Ho*Wo;
        }
    }

    thr_ptr = thrust::device_pointer_cast<float>(_b->_ptr);
    for (int i = 0; i < batch_size; ++i){
        for (int j = 0; j < C; ++j){
            thrust::fill(thr_ptr, thr_ptr + Ho*Wo, data_b[j]);
            thr_ptr += Ho*Wo;
        }
    }

    thr_ptr = thrust::device_pointer_cast<float>(_rm->_ptr);
    for (int i = 0; i < batch_size; ++i){
        for (int j = 0; j < C; ++j){
            thrust::fill(thr_ptr, thr_ptr + Ho*Wo, data_rm[j]);
            thr_ptr += Ho*Wo;
        }
    }

    thr_ptr = thrust::device_pointer_cast<float>(_rv->_ptr);
    for (int i = 0; i < batch_size; ++i){
        for (int j = 0; j < C; ++j){
            thrust::fill(thr_ptr, thr_ptr + Ho*Wo, std::sqrt(data_rv[j] + 1e-5f));
            thr_ptr += Ho*Wo;
        }
    }
}


Tensor<float>* BatchNormLayer::get_output()
{
    return _res;
}
