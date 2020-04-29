#ifndef CUDA_PROJ_LINEARLAYER_CUH
#define CUDA_PROJ_LINEARLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>

#include <cublas.h>
#include <cublas_v2.h>


class LinearLayer: public Layer {
public:
    LinearLayer(cublasHandle_t& cublas_handle_p, const std::string& w_path, int batch_size = 1);
    ~LinearLayer();

    void forward();

    void set_input(Tensor<float>* input);
    Tensor<float>* get_output();

private:
    Tensor<float>* _input, *_w, *_b, *_res, *_tmp;
    int batch_size, input_dim, output_dim;
    cublasHandle_t& cublas_handle;

};

#endif //CUDA_PROJ_LINEARLAYER_CUH
