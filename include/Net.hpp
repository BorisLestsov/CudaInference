#ifndef CUDA_PROJ_NET_CUH
#define CUDA_PROJ_NET_CUH

#include <vector>
#include <cublas.h>
#include <cublas_v2.h>
#include "helper_functions.cuh"

#include "Layer.hpp"
#include "Tensor.hpp"

class Net {
public:

    Net(cublasHandle_t& cublas_handle);

    void add_layer(Layer* layer);

    Tensor<float>* forward(Tensor<float>* data);


    std::vector<Layer*> layers;

private:

    cublasHandle_t& cublas_handle;

};




#endif //CUDA_PROJ_NET_CUH
