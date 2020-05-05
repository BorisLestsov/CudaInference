#include <stdexcept>
#include "AvgPoolLayer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"



AvgPoolLayer::AvgPoolLayer(int batch_size_p, int ker_size_p, int stride_p, int pad_p):
    batch_size(batch_size_p),
    _stride(stride_p),
    _pad(pad_p),
    H(ker_size_p),
    W(ker_size_p)
{
}

AvgPoolLayer::~AvgPoolLayer(){
    delete _res; 
}


__global__ 
void maxpool2d(float* src, float* res, int Hf, int Wf, int C, int Ho, int Wo, int Hi, int Wi, int batch_size, int stride, int pad, float pad_val=0)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int i_mat_stride = C*Hi*Wi;
    int o_mat_stride = C*Ho*Wo;
    int total_scals = C*Ho*Wo*batch_size;

    if (i >= total_scals){
        return;
    }
    
    int ni = i / (C*Ho*Wo);
    int ci = (i - ni*(C*Ho*Wo)) / (Ho*Wo);
    int hi = (i - ni*(C*Ho*Wo) - ci*(Ho*Wo)) / (Wo);
    int wi = (i - ni*(C*Ho*Wo) - ci*(Ho*Wo) - hi*(Wo));

    int Ri = hi * stride;
    int Rj = wi * stride;


    float resf = 0.0f;
    for (int ki = 0 ; ki < Hf; ++ki){
        for (int kj = 0 ; kj < Wf; ++kj){
            int inp_i = Ri + ki;
            int inp_j = Rj + kj;
            bool is_pad = (inp_i < pad) || (inp_j < pad) || (inp_i >= Hi + pad) || (inp_j >= Wi + pad);
            float el;
            // if (ni == 0 && ci ==0 && hi == 0 && wi == 0){
            //     printf("i=%d ; Ri=%d ; Rj=%d ; ci=%d ; hi=%d ; wi=%d ; ni=%d ; ki=%d ; kj=%d ; inp_i=%d ; inp_j=%d ; is_pad=%d\n", i, Ri, Rj, ci, hi, wi, ni, ki, kj, inp_i, inp_j, is_pad);
            //     ;
            // }
            if (is_pad){
                el = pad_val;
            }else{
                inp_i -= pad;
                inp_j -= pad;
                el = src[ni*i_mat_stride + ci*(Hi*Wi) + (inp_i)*(Wi) + inp_j];
            }

            resf += el; 
        }
    }

    res[ni*o_mat_stride + ci*(Ho*Wo) + hi*(Wo) + wi] = resf / (Hf*Wf);
}

void AvgPoolLayer::forward() 
{
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    int total = batch_size*C*Ho*Wo;
    num_blocks_x = total/cell_size + (total % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    maxpool2d<<<block_size, grid_size>>>(_input->_ptr, _res->_ptr, H, W, C, Ho, Wo, Hi, Wi, batch_size, _stride, _pad);
    debug_array(_res->_ptr, _res->count());

}


void AvgPoolLayer::set_input(Tensor<float>* input)
{
    if (input->size().size() != 4) {
        throw std::runtime_error("not four dims in input");
    }

    Size isize = input->size();
    //batch_size = isize[0];
    C = isize[1];
    Hi = isize[2];
    Wi = isize[3];
    Ho = (Hi + 2*_pad - 1*(H - 1) - 1)/_stride + 1;
    Wo = (Wi + 2*_pad - 1*(W - 1) - 1)/_stride + 1;

    if (input->size()[0] != batch_size) {
        throw std::runtime_error("batch size does not match");
    }
    //if (input->size()[1] != input_dim) {
    //    throw std::runtime_error(std::string("input dim is different: ") + std::to_string(input->size()[1]) + " vs " + std::to_string(input_dim));
    //}
    _input = input;

    _res = new Tensor<float>({batch_size, C, Ho, Wo});
}

Tensor<float>* AvgPoolLayer::get_output()
{
    return _res;
}
