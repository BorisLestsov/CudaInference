#include <stdexcept>
#include "ConvLayer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"
#include "npy.hpp"


__global__ void make_wcol(float* w_ptr, float* res_ptr, int N, int C, int H, int W)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int filt_stride = H*W;
    int mat_stride = C*H*W;

    if (i > N){
        return;
    }
    if (j > mat_stride){
        return;
    }

    int Ni = (i);
    int Ci = (j) / (H*W);
    int Hi = (j - Ci*H*W) / W ;
    int Wi = (j - Ci*H*W - Hi*W);

    res_ptr[i*mat_stride + j] = w_ptr[Ni*mat_stride + Ci*filt_stride + Hi*W + Wi];
}

__global__ void make_imcol(float* im_ptr, float* res_ptr, int Nf, int Cf, int Hf, int Wf, int Ho, int Wo, int Hi, int Wi, int batch_size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int filt_stride = Hf*Wf;
    int i_mat_stride = Cf*Hi*Wi;
    int total_scals = Ho*Wo*batch_size;
    int o_mat_stride = Cf*Hf*Wf;

    if (i >= total_scals){
        return;
    }
    if (j >= Hf*Wf*Cf){
        return;
    }

    int Ri = (i / Wo) % Hf;
    int Rj = (i % Wo) % Wf;
    int ci = j / (Hf*Wf);
    int K_ind_i = (j - ci*Hf*Wf) / Wf;
    int K_ind_j = (j - ci*Hf*Wf) % Wf;

    int hi = Ri + K_ind_i;
    int wi = Rj + K_ind_j;
    int ni = i / (Ho*Wo); // batch

    //printf("i=%d ; j=%d ; Ri=%d ; Rj=%d ; K_ind_i=%d ; K_ind_j=%d ; ci=%d ; hi=%d ; wi=%d ; ni=%d ; res=%d\n", i, j, Ri, Rj, K_ind_i, K_ind_j, ci, hi, wi, ni, i*o_mat_stride + j);

    res_ptr[i*o_mat_stride + j] = im_ptr[ni*i_mat_stride + ci*Hi*Wi + hi*Wi + wi];
}

#define Ndims 4
__global__ void transpose_ker(float* src_ptr, float* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int N = strides[0] * src_dims[0];
    if (i >= N){
        return;
    }

    int idx[Ndims];
    int new_idx[Ndims];
    int acc = 0;
    for (int k = 0; k < Ndims; ++k) {
        int cur_i = (i - acc) / strides[k];
        acc += cur_i*strides[k];

        idx[k] = cur_i;
        new_idx[reorder[k]] = cur_i;
    }

    // int Ni = i / strides[0];
    // int Ci = (i - Ni*strides[0]) / strides[1];
    // int Hi = (i - Ni*strides[0] - Ci*strides[1]) / strides[2];
    // int Wi = (i - Ni*strides[0] - Ci*strides[1] - Wi*strides[2]);

    int new_i = 0;
    for (int k = 0; k < Ndims; ++k) {
        new_i += new_strides[k]*new_idx[k];
    }

    dst_ptr[new_i] = src_ptr[i];
}


ConvLayer::ConvLayer(cublasHandle_t& cublas_handle, const std::string& w_path, int batch_size_p, int pad, int stride):
    cublas_handle(cublas_handle),
    _pad(pad),
    _stride(stride)
{
    batch_size = batch_size_p;

    std::vector<unsigned long> shape;
    std::vector<float> data;
    bool is_f;

    npy::LoadArrayFromNumpy(w_path + ".weight.npy", shape, is_f, data);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }


    N = 4;
    C = 3;
    H = 3;
    W = 3;
    _stride = 1;
    _pad = 0;

    _w = new Tensor<float>({N, C, H, W});
    _w->from_cpu(data.data());

    // npy::LoadArrayFromNumpy(w_path + ".bias.npy", shape, is_f, data);
    // if (is_f) {
    //     throw std::runtime_error("fortran format unsupported");
    // }
    // _b = new Tensor<float>({batch_size, output_dim});
    // float* tmp_ptr = _b->_ptr;
    // for (int i = 0; i < batch_size; ++i){
    //     cudaMemcpy(tmp_ptr, data.data(), output_dim*sizeof(float), cudaMemcpyHostToDevice);
    //     tmp_ptr += output_dim;
    // }

    //_tmp = new Tensor<float>({output_dim, batch_size});


    _wcol = new Tensor<float>({N, C*H*W});


    //std::cout << "INIT" << std::endl;

    // int cell_size = 32;
    // dim3 block_size;
    // dim3 grid_size;

    // int wcol_Ho = N;
    // int wcol_Wo = C*H*W;
    // int num_blocks_x = wcol_Ho/cell_size + (wcol_Ho % cell_size != 0);
    // int num_blocks_y = wcol_Wo/cell_size + (wcol_Wo % cell_size != 0);
    // block_size = dim3(cell_size, cell_size);
    // grid_size = dim3(num_blocks_x, num_blocks_y, 3);

    // make_wcol<<<block_size, grid_size>>>(_w->_ptr, _wcol->_ptr, N, C, H, W);

}

void ConvLayer::forward() 
{
    _imcol = new Tensor<float>({C*H*W, batch_size*Ho*Wo});

    //_res = new Tensor<float>({batch_size, C, Ho, Wo});
    _res = new Tensor<float>({N, batch_size*Ho*Wo});
    _tmp = new Tensor<float>({N, batch_size*Ho*Wo});


    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;
    int num_blocks_y;

    int imcol_Ho = batch_size*Ho*Wo;
    int imcol_Wo = C*H*W;
    num_blocks_x = imcol_Ho/cell_size + (imcol_Ho % cell_size != 0);
    num_blocks_y = imcol_Wo/cell_size + (imcol_Wo % cell_size != 0);
    block_size = dim3(cell_size, cell_size);
    grid_size = dim3(num_blocks_x, num_blocks_y);

    make_imcol<<<block_size, grid_size>>>(_input->_ptr, _imcol->_ptr, N, C, H, W, Ho, Wo, Hi, Wi, batch_size);
    //debug_array(_imcol->_ptr, _imcol->count());

    // make_wcol:
    _wcol = _w;

    int m = N;
    int n = batch_size*Ho*Wo;
    int k = C*H*W;
    row_major_sgemm(cublas_handle, m, n, k, _wcol->_ptr, _imcol->_ptr, _res->_ptr, _tmp->_ptr);

    num_blocks_x = (N*n)/cell_size + ((N*n) % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    Size dims_cpu({N, batch_size, Ho, Wo});
    _dims = new Tensor<int>({4});
    _dims->from_cpu(dims_cpu.data());

    Size strides_cpu({batch_size*Ho*Wo, Ho*Wo, Wo, 1});
    _strides = new Tensor<int>({4});
    _strides->from_cpu(strides_cpu.data());

    Size reorder_cpu({1, 0, 2, 3});
    _reorder = new Tensor<int>({4});
    _reorder->from_cpu(reorder_cpu.data());

    Size new_strides_cpu({N*Ho*Wo, Ho*Wo, Wo, 1});
    _new_strides = new Tensor<int>({4});
    _new_strides->from_cpu(new_strides_cpu.data());

    transpose_ker<<<grid_size, block_size>>>(_res->_ptr, _tmp->_ptr, _dims->_ptr, _strides->_ptr, _reorder->_ptr, _new_strides->_ptr);
    _tmp->reshape({batch_size, N, Ho, Wo});

    debug_array(_tmp->_ptr, _tmp->count());
}

ConvLayer::~ConvLayer(){

    delete _w; 
    //delete _b; 
    delete _imcol;
    delete _wcol;

    delete _res; 
    delete _tmp;

    delete _dims;
    delete _reorder;
    delete _strides;
    delete _new_strides;
}

void ConvLayer::set_input(Tensor<float>* input)
{
    if (input->size().size() != 4) {
        throw std::runtime_error("not four dims in input");
    }

    Size isize = input->size();
    //batch_size = isize[0];
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
}

Tensor<float>* ConvLayer::get_output()
{
    return _res;
}

int ConvLayer::get_output_dim()
{
    //return output_dim;
    return 0;
}
