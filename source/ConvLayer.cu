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

__global__ void make_imcol(float* im_ptr, float* res_ptr, int Nf, int Cf, int Hf, int Wf, int Ho, int Wo, int Hi, int Wi, int batch_size, int pad, float pad_val=0)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    //int filt_stride = Hf*Wf;
    int i_mat_stride = Cf*Hi*Wi;
    int total_scals = Ho*Wo*batch_size;
    int o_mat_stride = Cf*Hf*Wf;

    if (i >= total_scals){
        return;
    }
    if (j >= Hf*Wf*Cf){
        return;
    }

    int Ri = (i / Wo);
    int Rj = (i % Wo);
    int ci = j / (Hf*Wf);
    int K_ind_i = (j - ci*Hf*Wf) / Wf;
    int K_ind_j = (j - ci*Hf*Wf) % Wf;

    int hi = Ri + K_ind_i;
    int wi = Rj + K_ind_j;
    int ni = i / (Ho*Wo); // batch


    bool is_pad = (hi < pad) || (wi < pad) || (hi >= Hi + pad) || (wi >= Wi + pad);
    if (i == 3){
        //printf("i=%d ; j=%d ; Ri=%d ; Rj=%d ; K_ind_i=%d ; K_ind_j=%d ; ci=%d ; hi=%d ; wi=%d ; ni=%d ; res=%d ; is_pad=%d\n", i, j, Ri, Rj, K_ind_i, K_ind_j, ci, hi, wi, ni, i*o_mat_stride + j, is_pad);
        ;
    }

    if (is_pad){
        res_ptr[i*o_mat_stride + j] = pad_val;
        return;
    }
    hi -= pad;
    wi -= pad;
    res_ptr[i*o_mat_stride + j] = im_ptr[ni*i_mat_stride + ci*Hi*Wi + hi*Wi + wi];
}

#define Ndims 4
__global__ void transpose_ker(float* src_ptr, float* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int total = strides[0] * src_dims[0];
    if (i >= total){
        return;
    }

    //int idx[Ndims];
    int new_idx[Ndims];
    int acc = 0;
    for (int k = 0; k < Ndims; ++k) {
        int cur_i = (i - acc) / strides[k];
        acc += cur_i*strides[k];

        //idx[k] = cur_i;
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

    //printf("(%d %d %d %d)  ->  (%d %d %d %d)       %d %d      %d %d %d %d \n", idx[0], idx[1], idx[2], idx[3], new_idx[0], new_idx[1], new_idx[2], new_idx[3], i, new_i, src_dims[0], src_dims[1], src_dims[2], src_dims[3]);

    dst_ptr[new_i] = src_ptr[i];
}


ConvLayer::ConvLayer(cublasHandle_t& cublas_handle, const std::string& w_path, int batch_size_p, int pad, int stride, bool bias):
    cublas_handle(cublas_handle),
    _pad(pad),
    _stride(stride),
    _bias(bias),
    input_set(false)
{
    batch_size = batch_size_p;

    std::vector<unsigned long> shape;
    std::vector<float> data;
    bool is_f;

    npy::LoadArrayFromNumpy(w_path + ".weight.npy", shape, is_f, data);
    if (is_f) {
        throw std::runtime_error("fortran format unsupported");
    }

    N = shape[0];
    C = shape[1];
    H = shape[2];
    W = shape[3];

    _w = new Tensor<float>({N, C, H, W});
    _w->from_cpu(data.data());


    if (_bias){
        std::vector<unsigned long> shape_b;
        npy::LoadArrayFromNumpy(w_path + ".bias.npy", shape_b, is_f, data_b);
        if (is_f) {
            throw std::runtime_error("fortran format unsupported");
        }
    }


    _wcol = new Tensor<float>({N, C*H*W});

    bool weights_nchw = true;
    if (weights_nchw) {
        // weights already in im2col format
        _wcol = _w;
    } else {
        // transform weights to im2col format
        int cell_size = 32;
        dim3 block_size;
        dim3 grid_size;

        int wcol_Ho = N;
        int wcol_Wo = C*H*W;
        int num_blocks_x = wcol_Ho/cell_size + (wcol_Ho % cell_size != 0);
        int num_blocks_y = wcol_Wo/cell_size + (wcol_Wo % cell_size != 0);
        block_size = dim3(cell_size, cell_size);
        grid_size = dim3(num_blocks_x, num_blocks_y, 3);

        make_wcol<<<block_size, grid_size>>>(_w->_ptr, _wcol->_ptr, N, C, H, W);
    }
}

void ConvLayer::forward() 
{
    if (! input_set){
        throw std::runtime_error("input not set in forward");
    }

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

    make_imcol<<<block_size, grid_size>>>(_input->_ptr, _imcol->_ptr, N, C, H, W, Ho, Wo, Hi, Wi, batch_size, _pad);
    //debug_array(_imcol->_ptr, _imcol->count());


    row_major_sgemm(cublas_handle, m, n, k, _wcol->_ptr, _imcol->_ptr, _res->_ptr, _tmp->_ptr);

    num_blocks_x = (N*n)/cell_size + ((N*n) % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    transpose_ker<<<grid_size, block_size>>>(_res->_ptr, _tmp->_ptr, _dims->_ptr, _strides->_ptr, _reorder->_ptr, _new_strides->_ptr);
    _tmp->reshape({batch_size, N, Ho, Wo});

    if (_bias) {
        Tensor<float>::add_inplace(_tmp, _bcol);
    }

    debug_array(_tmp->_ptr, _tmp->count());
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
    m = N;
    n = batch_size*Ho*Wo;
    k = C*H*W;

    if (input->size()[0] != batch_size) {
        throw std::runtime_error("batch size does not match");
    }
    //if (input->size()[1] != input_dim) {
    //    throw std::runtime_error(std::string("input dim is different: ") + std::to_string(input->size()[1]) + " vs " + std::to_string(input_dim));
    //}
    _input = input;

    _imcol = new Tensor<float>({C*H*W, batch_size*Ho*Wo});

    //_res = new Tensor<float>({batch_size, C, Ho, Wo});
    _res = new Tensor<float>({N, batch_size*Ho*Wo});
    _tmp = new Tensor<float>({N, batch_size*Ho*Wo});



    // create arrays for reshape

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

    if (_bias) {
        // bias array to add
        _bcol = new Tensor<float>({batch_size, N, Ho, Wo});
        thrust::device_ptr<float> thr_ptr = thrust::device_pointer_cast<float>(_bcol->_ptr);
        for (int i = 0; i < batch_size; ++i){
            for (int j = 0; j < N; ++j){
                thrust::fill(thr_ptr, thr_ptr + Ho*Wo, data_b[j]);
                thr_ptr += Ho*Wo;
            }
        }
    }
    input_set = true;
}

ConvLayer::~ConvLayer(){

    delete _w; 
    //delete _b; 
    delete _imcol;
    delete _wcol;
    if (_bias) {
        delete _bcol; 
    }

    delete _res; 
    delete _tmp;

    delete _dims;
    delete _reorder;
    delete _strides;
    delete _new_strides;
}

Tensor<float>* ConvLayer::get_output()
{
    return _tmp;
}

int ConvLayer::get_output_dim()
{
    //return output_dim;
    return 0;
}
