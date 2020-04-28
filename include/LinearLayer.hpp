#ifndef CUDA_PROJ_LINEARLAYER_CUH
#define CUDA_PROJ_LINEARLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <cublas.h>
#include <cublas_v2.h>


class LinearLayer: public Layer {
public:
    LinearLayer();

    void forward();

    void forward_tmp(cublasHandle_t& cublas_handle, Tensor<float>* input);

private:
    float* _w, *_res, *_tmp;

};

#endif //CUDA_PROJ_LINEARLAYER_CUH
