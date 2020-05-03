#ifndef CUDA_PROJ_CONVLAYER_CUH
#define CUDA_PROJ_CONVLAYER_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <string>

#include <cublas.h>
#include <cublas_v2.h>


class ConvLayer: public Layer {
public:
    ConvLayer(cublasHandle_t& cublas_handle_p, const std::string& w_path, int batch_size = 1, int stride=1, int pad=0);
    ~ConvLayer();

    void forward();

    void set_input(Tensor<float>* input);
    Tensor<float>* get_output();
    int get_output_dim();

private:
    Tensor<float>* _input, *_w, *_b, *_res;
    Tensor<float>* _imcol, *_wcol, *_tmp;
    Tensor<int>* _dims, *_reorder, *_strides, *_new_strides;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int batch_size, N, C, H, W, _pad, _stride;
    cublasHandle_t& cublas_handle;

};

#endif //CUDA_PROJ_CONVLAYER_CUH
